"""
database/mongodb.py
MongoDB Atlas integration for MedSecure AI
Handles all CRUD operations for medicine reports
"""

import os
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class MedSecureDB:
    """MongoDB Atlas database handler for MedSecure AI."""

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._connect()

    def _connect(self):
        """Establish connection to MongoDB Atlas."""
        try:
            uri = os.getenv(
                "MONGODB_URI",
                "mongodb://localhost:27017/medsecure_ai"  # fallback for local dev
            )
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Ping to verify connection
            self.client.admin.command("ping")
            self.db = self.client["medsecure_ai"]
            self.collection = self.db["medicine_reports"]
            # Create indexes for fast lookups
            self.collection.create_index([("timestamp", DESCENDING)])
            self.collection.create_index([("medicine_name", 1)])
            self.collection.create_index([("prediction", 1)])
            logger.info("✅ Connected to MongoDB Atlas successfully.")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            # App will still run; DB ops will return graceful errors
            self.client = None

    def is_connected(self):
        """Check if DB connection is alive."""
        return self.client is not None

    # ─────────────────────────────────────────────
    # CREATE
    # ─────────────────────────────────────────────
    def insert_report(self, report: dict) -> str | None:
        """
        Insert a new medicine analysis report.

        Args:
            report (dict): Report document to insert.

        Returns:
            str: Inserted document ID as string, or None on failure.
        """
        if not self.is_connected():
            logger.warning("DB not connected. Skipping insert.")
            return None
        try:
            report["timestamp"] = datetime.utcnow()
            result = self.collection.insert_one(report)
            logger.info(f"Report inserted: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return None

    # ─────────────────────────────────────────────
    # READ – single record
    # ─────────────────────────────────────────────
    def get_report_by_id(self, report_id: str) -> dict | None:
        """Fetch a single report by its MongoDB ObjectId string."""
        if not self.is_connected():
            return None
        try:
            from bson import ObjectId
            doc = self.collection.find_one({"_id": ObjectId(report_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as e:
            logger.error(f"Fetch by ID failed: {e}")
            return None

    # ─────────────────────────────────────────────
    # READ – all records (history)
    # ─────────────────────────────────────────────
    def get_all_reports(self, limit: int = 50) -> list[dict]:
        """
        Retrieve the most recent reports, newest first.

        Args:
            limit (int): Maximum records to return.

        Returns:
            list[dict]: List of report documents.
        """
        if not self.is_connected():
            return []
        try:
            docs = list(
                self.collection.find().sort("timestamp", DESCENDING).limit(limit)
            )
            for d in docs:
                d["_id"] = str(d["_id"])
                if "timestamp" in d:
                    d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            return docs
        except Exception as e:
            logger.error(f"Get all reports failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────
    def search_reports(self, query: str) -> list[dict]:
        """
        Search reports by medicine name or manufacturer (case-insensitive).

        Args:
            query (str): Search term.

        Returns:
            list[dict]: Matching report documents.
        """
        if not self.is_connected():
            return []
        try:
            regex = {"$regex": query, "$options": "i"}
            docs = list(
                self.collection.find(
                    {"$or": [{"medicine_name": regex}, {"manufacturer": regex}]}
                ).sort("timestamp", DESCENDING).limit(20)
            )
            for d in docs:
                d["_id"] = str(d["_id"])
                if "timestamp" in d:
                    d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            return docs
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────
    def get_statistics(self) -> dict:
        """
        Aggregate statistics for the dashboard.

        Returns:
            dict: Summary stats (total, genuine, counterfeit, avg score).
        """
        if not self.is_connected():
            return {"total": 0, "genuine": 0, "counterfeit": 0, "avg_score": 0}
        try:
            total = self.collection.count_documents({})
            genuine = self.collection.count_documents({"prediction": "Genuine"})
            counterfeit = self.collection.count_documents({"prediction": "Counterfeit"})
            pipeline = [{"$group": {"_id": None, "avg": {"$avg": "$authenticity_score"}}}]
            avg_result = list(self.collection.aggregate(pipeline))
            avg_score = round(avg_result[0]["avg"], 1) if avg_result else 0
            return {
                "total": total,
                "genuine": genuine,
                "counterfeit": counterfeit,
                "avg_score": avg_score,
            }
        except Exception as e:
            logger.error(f"Statistics fetch failed: {e}")
            return {"total": 0, "genuine": 0, "counterfeit": 0, "avg_score": 0}

    # ─────────────────────────────────────────────
    # DELETE (admin / cleanup)
    # ─────────────────────────────────────────────
    def delete_report(self, report_id: str) -> bool:
        """Delete a report by ID."""
        if not self.is_connected():
            return False
        try:
            from bson import ObjectId
            result = self.collection.delete_one({"_id": ObjectId(report_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False


# Singleton instance used across the application
db = MedSecureDB()
