import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ManifestManager:
    """Tracks MD5 hashes of parsed JSON files to support incremental embedding."""
    def __init__(self, manifest_dir: Path):
        self.manifest_file = manifest_dir / "embed_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embed manifest: {e}")
        return {}

    def _save_manifest(self):
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def is_file_changed(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
            
        file_hash = self._compute_hash(file_path)
        file_key = str(file_path.name)
        
        return self.manifest.get(file_key) != file_hash

    def update_file(self, file_path: Path):
        if not file_path.exists():
            return
            
        file_hash = self._compute_hash(file_path)
        file_key = str(file_path.name)
        
        self.manifest[file_key] = file_hash
        self._save_manifest()

    def _compute_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
