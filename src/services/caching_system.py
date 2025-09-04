"""
Advanced Caching System for Transcriptions and Translations
Provides intelligent caching with content-based keys, compression, and automatic cleanup.
"""

import os
import json
import pickle
import gzip
import hashlib
import sqlite3
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cache entry"""
    key: str
    content_hash: str
    data_type: str  # 'transcription', 'translation', 'audio_analysis'
    data: Any
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    model_version: Optional[str] = None
    language: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 1
    compressed: bool = False
    tags: List[str] = field(default_factory=list)

class CacheStats:
    """Cache statistics and metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entry_count = 0
        
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size_bytes': self.size_bytes,
            'entry_count': self.entry_count,
            'hit_rate': self.hit_rate
        }

class ContentHasher:
    """Generates content-based cache keys"""
    
    @staticmethod
    def hash_file(file_path: Path, chunk_size: int = 8192) -> str:
        """Generate hash for file content"""
        hasher = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
                    
            file_stat = file_path.stat()
            # Include file metadata in hash for additional uniqueness
            metadata = f"{file_stat.st_size}:{file_stat.st_mtime}"
            hasher.update(metadata.encode())
            
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            # Fallback to path and timestamp
            hasher.update(str(file_path).encode())
            hasher.update(str(datetime.utcnow().timestamp()).encode())
            
        return hasher.hexdigest()
    
    @staticmethod
    def hash_content(content: str, **kwargs) -> str:
        """Generate hash for text content with optional parameters"""
        hasher = hashlib.sha256()
        hasher.update(content.encode('utf-8'))
        
        # Include relevant parameters in hash
        for key, value in sorted(kwargs.items()):
            if value is not None:
                hasher.update(f"{key}:{value}".encode())
                
        return hasher.hexdigest()

class CacheStorage:
    """Handles physical storage of cache data"""
    
    def __init__(self, cache_dir: Path, use_compression: bool = True):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression
        self.lock = threading.RLock()
        
    def _get_file_path(self, key: str, compressed: bool = False) -> Path:
        """Get file path for cache key"""
        ext = '.pkl.gz' if compressed else '.pkl'
        return self.cache_dir / f"{key}{ext}"
    
    def save(self, key: str, data: Any, compress: bool = None) -> Tuple[Path, bool]:
        """Save data to storage"""
        compress = compress if compress is not None else self.use_compression
        file_path = self._get_file_path(key, compress)
        
        with self.lock:
            try:
                serialized = pickle.dumps(data)
                
                if compress:
                    with gzip.open(file_path, 'wb') as f:
                        f.write(serialized)
                else:
                    with open(file_path, 'wb') as f:
                        f.write(serialized)
                        
                return file_path, compress
                
            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")
                raise
                
    def load(self, key: str, compressed: bool = None) -> Optional[Any]:
        """Load data from storage"""
        # Try both compressed and uncompressed
        for comp in [compressed, True, False] if compressed is None else [compressed]:
            if comp is None:
                continue
                
            file_path = self._get_file_path(key, comp)
            
            if not file_path.exists():
                continue
                
            with self.lock:
                try:
                    if comp:
                        with gzip.open(file_path, 'rb') as f:
                            return pickle.load(f)
                    else:
                        with open(file_path, 'rb') as f:
                            return pickle.load(f)
                            
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key} (compressed={comp}): {e}")
                    continue
                    
        return None
    
    def delete(self, key: str):
        """Delete cache entry from storage"""
        with self.lock:
            for compressed in [True, False]:
                file_path = self._get_file_path(key, compressed)
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file_path}: {e}")
    
    def get_size(self, key: str) -> int:
        """Get size of cache entry on disk"""
        total_size = 0
        
        for compressed in [True, False]:
            file_path = self._get_file_path(key, compressed)
            if file_path.exists():
                total_size += file_path.stat().st_size
                
        return total_size
    
    def cleanup_orphaned_files(self, valid_keys: set):
        """Remove cache files that don't have corresponding metadata entries"""
        with self.lock:
            for file_path in self.cache_dir.glob("*.pkl*"):
                key = file_path.stem.split('.')[0]  # Remove .pkl or .pkl.gz
                if key not in valid_keys:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed orphaned cache file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove orphaned file {file_path}: {e}")

class CacheDatabase:
    """SQLite database for cache metadata"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        file_path TEXT,
                        file_size INTEGER,
                        model_version TEXT,
                        language TEXT,
                        created_at TIMESTAMP NOT NULL,
                        accessed_at TIMESTAMP NOT NULL,
                        access_count INTEGER DEFAULT 1,
                        compressed BOOLEAN DEFAULT FALSE,
                        tags TEXT,  -- JSON array
                        size_bytes INTEGER DEFAULT 0
                    )
                ''')
                
                conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON cache_entries(content_hash)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON cache_entries(data_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)')
                
                conn.commit()
            finally:
                conn.close()
    
    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry metadata"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    SELECT * FROM cache_entries WHERE key = ?
                ''', (key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                    
                tags = json.loads(row[11]) if row[11] else []
                
                return CacheEntry(
                    key=row[0],
                    content_hash=row[1],
                    data_type=row[2],
                    data=None,  # Data loaded separately
                    file_path=row[3],
                    file_size=row[4],
                    model_version=row[5],
                    language=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    accessed_at=datetime.fromisoformat(row[8]),
                    access_count=row[9],
                    compressed=bool(row[10]),
                    tags=tags
                )
                
            finally:
                conn.close()
    
    def save_entry(self, entry: CacheEntry, size_bytes: int = 0):
        """Save cache entry metadata"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, content_hash, data_type, file_path, file_size, model_version, 
                     language, created_at, accessed_at, access_count, compressed, tags, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.key,
                    entry.content_hash,
                    entry.data_type,
                    entry.file_path,
                    entry.file_size,
                    entry.model_version,
                    entry.language,
                    entry.created_at.isoformat(),
                    entry.accessed_at.isoformat(),
                    entry.access_count,
                    entry.compressed,
                    json.dumps(entry.tags),
                    size_bytes
                ))
                conn.commit()
            finally:
                conn.close()
    
    def update_access(self, key: str):
        """Update access time and count"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    UPDATE cache_entries 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE key = ?
                ''', (datetime.utcnow().isoformat(), key))
                conn.commit()
            finally:
                conn.close()
    
    def delete_entry(self, key: str):
        """Delete cache entry metadata"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                conn.commit()
            finally:
                conn.close()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    SELECT COUNT(*), SUM(size_bytes), SUM(access_count)
                    FROM cache_entries
                ''')
                
                row = cursor.fetchone()
                stats = CacheStats()
                
                if row and row[0]:
                    stats.entry_count = row[0]
                    stats.size_bytes = row[1] or 0
                    # Access count serves as total operations for hit rate calculation
                    
                return stats
                
            finally:
                conn.close()
    
    def get_entries_by_age(self, older_than: timedelta) -> List[str]:
        """Get cache keys older than specified age"""
        cutoff_time = datetime.utcnow() - older_than
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    SELECT key FROM cache_entries 
                    WHERE accessed_at < ?
                    ORDER BY accessed_at ASC
                ''', (cutoff_time.isoformat(),))
                
                return [row[0] for row in cursor.fetchall()]
                
            finally:
                conn.close()
    
    def get_all_keys(self) -> set:
        """Get all cache keys"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('SELECT key FROM cache_entries')
                return {row[0] for row in cursor.fetchall()}
            finally:
                conn.close()

class AdvancedCache:
    """Advanced caching system with intelligent eviction and compression"""
    
    def __init__(self, 
                 cache_dir: Path = None,
                 max_size_gb: float = 2.0,
                 max_age_days: int = 30,
                 use_compression: bool = True):
        
        self.cache_dir = cache_dir or Path.home() / ".cache" / "transcription_tool"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.max_age = timedelta(days=max_age_days)
        
        self.storage = CacheStorage(self.cache_dir, use_compression)
        self.database = CacheDatabase(self.cache_dir / "cache.db")
        self.hasher = ContentHasher()
        
        self.stats = CacheStats()
        self.lock = asyncio.Lock()
        
        # Periodic cleanup
        self._cleanup_task = None
        
    async def start(self):
        """Start cache maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
    async def stop(self):
        """Stop cache maintenance tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    @asynccontextmanager
    async def cache_context(self):
        """Context manager for cache lifecycle"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    def _generate_key(self, content_hash: str, data_type: str, **kwargs) -> str:
        """Generate cache key from content hash and parameters"""
        key_parts = [content_hash, data_type]
        
        # Add relevant parameters to key
        for param in ['model_version', 'language', 'task']:
            if param in kwargs and kwargs[param]:
                key_parts.append(f"{param}={kwargs[param]}")
                
        return "_".join(key_parts)
    
    async def get_transcription(self, 
                              file_path: Path, 
                              model_version: str = None,
                              language: str = None) -> Optional[Dict[str, Any]]:
        """Get cached transcription"""
        
        content_hash = self.hasher.hash_file(file_path)
        key = self._generate_key(
            content_hash, 
            'transcription', 
            model_version=model_version,
            language=language
        )
        
        return await self._get(key)
    
    async def set_transcription(self,
                              file_path: Path,
                              transcription_data: Dict[str, Any],
                              model_version: str = None,
                              language: str = None):
        """Cache transcription result"""
        
        content_hash = self.hasher.hash_file(file_path)
        key = self._generate_key(
            content_hash,
            'transcription',
            model_version=model_version, 
            language=language
        )
        
        entry = CacheEntry(
            key=key,
            content_hash=content_hash,
            data_type='transcription',
            data=transcription_data,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            model_version=model_version,
            language=language,
            tags=['transcription', model_version or 'unknown_model']
        )
        
        await self._set(key, entry)
    
    async def get_translation(self,
                            text: str,
                            target_language: str,
                            source_language: str = None) -> Optional[str]:
        """Get cached translation"""
        
        content_hash = self.hasher.hash_content(
            text,
            target_language=target_language,
            source_language=source_language
        )
        key = self._generate_key(
            content_hash,
            'translation',
            target_language=target_language,
            source_language=source_language
        )
        
        result = await self._get(key)
        return result.get('translated_text') if result else None
    
    async def set_translation(self,
                            text: str,
                            translated_text: str,
                            target_language: str,
                            source_language: str = None):
        """Cache translation result"""
        
        content_hash = self.hasher.hash_content(
            text,
            target_language=target_language,
            source_language=source_language
        )
        key = self._generate_key(
            content_hash,
            'translation',
            target_language=target_language,
            source_language=source_language
        )
        
        translation_data = {
            'original_text': text,
            'translated_text': translated_text,
            'target_language': target_language,
            'source_language': source_language,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        entry = CacheEntry(
            key=key,
            content_hash=content_hash,
            data_type='translation',
            data=translation_data,
            language=f"{source_language or 'auto'}->{target_language}",
            tags=['translation', target_language]
        )
        
        await self._set(key, entry)
    
    async def _get(self, key: str) -> Optional[Any]:
        """Internal get method"""
        async with self.lock:
            # Check metadata
            entry = self.database.get_entry(key)
            if not entry:
                self.stats.misses += 1
                return None
            
            # Load data from storage
            data = self.storage.load(key, entry.compressed)
            if data is None:
                # Metadata exists but data is missing, cleanup
                self.database.delete_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access statistics
            self.database.update_access(key)
            self.stats.hits += 1
            
            return data
    
    async def _set(self, key: str, entry: CacheEntry):
        """Internal set method"""
        async with self.lock:
            try:
                # Save data to storage
                file_path, compressed = self.storage.save(key, entry.data)
                entry.compressed = compressed
                
                # Get size
                size_bytes = self.storage.get_size(key)
                
                # Save metadata
                self.database.save_entry(entry, size_bytes)
                
                # Update stats
                self.stats.entry_count += 1
                self.stats.size_bytes += size_bytes
                
                # Check if cleanup is needed
                if self.stats.size_bytes > self.max_size_bytes:
                    await self._cleanup_by_size()
                    
            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
    
    async def _cleanup_by_size(self):
        """Cleanup cache by size limit"""
        if self.stats.size_bytes <= self.max_size_bytes:
            return
            
        logger.info("Cache size limit exceeded, performing cleanup")
        
        # Get entries sorted by access time (LRU)
        target_size = int(self.max_size_bytes * 0.8)  # Clean to 80% of limit
        
        old_entries = self.database.get_entries_by_age(timedelta(days=0))  # All entries
        removed_size = 0
        removed_count = 0
        
        for key in old_entries:
            if self.stats.size_bytes - removed_size <= target_size:
                break
                
            entry_size = self.storage.get_size(key)
            
            # Remove entry
            self.storage.delete(key)
            self.database.delete_entry(key)
            
            removed_size += entry_size
            removed_count += 1
        
        self.stats.size_bytes -= removed_size
        self.stats.entry_count -= removed_count
        self.stats.evictions += removed_count
        
        logger.info(f"Cleaned up {removed_count} entries, freed {removed_size / 1024**2:.1f}MB")
    
    async def _cleanup_by_age(self):
        """Cleanup cache by age"""
        old_keys = self.database.get_entries_by_age(self.max_age)
        
        if not old_keys:
            return
            
        logger.info(f"Removing {len(old_keys)} expired cache entries")
        
        removed_size = 0
        for key in old_keys:
            removed_size += self.storage.get_size(key)
            self.storage.delete(key)
            self.database.delete_entry(key)
        
        self.stats.size_bytes -= removed_size
        self.stats.entry_count -= len(old_keys)
        self.stats.evictions += len(old_keys)
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                await self._cleanup_by_age()
                
                # Cleanup orphaned files
                valid_keys = self.database.get_all_keys()
                self.storage.cleanup_orphaned_files(valid_keys)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def clear_all(self):
        """Clear all cache entries"""
        async with self.lock:
            all_keys = self.database.get_all_keys()
            
            for key in all_keys:
                self.storage.delete(key)
                self.database.delete_entry(key)
            
            self.stats = CacheStats()
            logger.info("Cache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        db_stats = self.database.get_stats()
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': self.stats.hit_rate,
            'entry_count': db_stats.entry_count,
            'size_bytes': db_stats.size_bytes,
            'size_mb': db_stats.size_bytes / 1024**2,
            'evictions': self.stats.evictions,
            'cache_dir': str(self.cache_dir)
        }

# Global cache instance
cache_system = AdvancedCache()