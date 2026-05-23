import threading
import uuid
import time
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 3600  # Completed/failed jobs are purged after 1 hour

class AnalysisJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending" # pending, running, completed, failed
        self.events: List[Dict[str, Any]] = []
        self.result = None
        self.error = None
        self._lock = threading.Lock()
        
        self.add_event("pending", "Job initialized and waiting to start...")

    def add_event(self, status: str, message: str, data: Any = None):
        with self._lock:
            self.status = status
            event = {
                "timestamp": time.time(),
                "status": status,
                "message": message
            }
            if data:
                event["data"] = data
            self.events.append(event)
            logger.info(f"[Job {self.job_id}] {message}")

    def complete(self, result: Any):
        with self._lock:
            self.status = "completed"
            self.result = result
            event = {
                "timestamp": time.time(),
                "status": "completed",
                "message": "Analysis completed successfully",
                "result_ready": True
            }
            self.events.append(event)
            logger.info(f"[Job {self.job_id}] Completed successfully")

    def fail(self, error: str):
        with self._lock:
            self.status = "failed"
            self.error = error
            event = {
                "timestamp": time.time(),
                "status": "failed",
                "message": f"Analysis failed: {error}",
                "error": error
            }
            self.events.append(event)
            logger.error(f"[Job {self.job_id}] Failed: {error}")

    def get_events_since(self, index: int) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.events[index:])

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "events_count": len(self.events),
                "error": self.error,
                "latest_message": self.events[-1]["message"] if self.events else ""
            }


class JobStore:
    def __init__(self):
        self.jobs: Dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()
        # Start background cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Periodically remove completed/failed jobs older than JOB_TTL_SECONDS."""
        def _cleanup():
            while True:
                time.sleep(300)  # Run every 5 minutes
                now = time.time()
                with self._lock:
                    to_delete = [
                        job_id for job_id, job in self.jobs.items()
                        if job.status in ("completed", "failed")
                        and job.events
                        and (now - job.events[-1]["timestamp"]) > JOB_TTL_SECONDS
                    ]
                    for job_id in to_delete:
                        del self.jobs[job_id]
                        logger.debug(f"[JobStore] Evicted expired job {job_id}")
        t = threading.Thread(target=_cleanup, daemon=True, name="job-store-cleanup")
        t.start()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self.jobs[job_id] = AnalysisJob(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        with self._lock:
            return self.jobs.get(job_id)


# Global job store instance
analysis_jobs = JobStore()
