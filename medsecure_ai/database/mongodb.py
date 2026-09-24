"""
database/mongodb.py
Local MongoDB integration for MedSecure AI
Handles all CRUD operations for medicine reports
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict

from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)


class MedSecureDB:
    """Local MongoDB database handler for MedSecure AI."""

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._connect()

    # ─────────────────────────────────────────────
    # DATABASE CONNECTION
    # ─────────────────────────────────────────────

    def _connect(self):
        """Establish connection to local MongoDB."""

        try:
            # Get MongoDB URI from .env
            # If not found, use local MongoDB as fallback
            uri = os.getenv(
                "MONGODB_URI",
                "mongodb://localhost:27017/"
            )

            logger.info(f"Connecting to MongoDB: {uri}")

            # Create MongoDB client
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000
            )

            # Ping MongoDB to verify connection
            self.client.admin.command("ping")

            # Select database
            self.db = self.client["medsecure_ai"]

            # Select collection
            self.collection = self.db["medicine_reports"]

            # Create indexes for faster queries
            self.collection.create_index(
                [("timestamp", DESCENDING)]
            )

            self.collection.create_index(
                [("medicine_name", 1)]
            )

            self.collection.create_index(
                [("prediction", 1)]
            )

            logger.info(
                "✅ Connected to local MongoDB successfully."
            )

        except (
            ConnectionFailure,
            ServerSelectionTimeoutError
        ) as e:

            logger.error(
                f"❌ MongoDB connection failed: {e}"
            )

            # Keep application running even if DB is unavailable
            self.client = None
            self.db = None
            self.collection = None

    # ─────────────────────────────────────────────
    # CONNECTION STATUS
    # ─────────────────────────────────────────────

    def is_connected(self):
        """Check if MongoDB connection is alive."""

        return self.client is not None

    # ─────────────────────────────────────────────
    # CREATE
    # ─────────────────────────────────────────────

    def insert_report(self, report: dict) -> Optional[str]:
        """
        Insert a new medicine analysis report.

        Args:
            report (dict): Report document to insert.

        Returns:
            str: Inserted document ID as string.
            None if insertion fails.
        """

        if not self.is_connected():
            logger.warning(
                "DB not connected. Skipping insert."
            )
            return None

        try:

            # Add timestamp
            report["timestamp"] = datetime.utcnow()

            # Insert document
            result = self.collection.insert_one(report)

            logger.info(
                f"Report inserted: {result.inserted_id}"
            )

            return str(result.inserted_id)

        except Exception as e:

            logger.error(
                f"Insert failed: {e}"
            )

            return None

    # ─────────────────────────────────────────────
    # READ - SINGLE RECORD
    # ─────────────────────────────────────────────

    def get_report_by_id(
        self,
        report_id: str
    ) -> Optional[dict]:

        """
        Fetch a single report by MongoDB ObjectId string.
        """

        if not self.is_connected():
            return None

        try:

            from bson import ObjectId

            doc = self.collection.find_one(
                {"_id": ObjectId(report_id)}
            )

            if doc:

                doc["_id"] = str(doc["_id"])

                if "timestamp" in doc:
                    doc["timestamp"] = doc[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            return doc

        except Exception as e:

            logger.error(
                f"Fetch by ID failed: {e}"
            )

            return None

    # ─────────────────────────────────────────────
    # READ - ALL REPORTS
    # ─────────────────────────────────────────────

    def get_all_reports(
        self,
        limit: int = 50,
        prediction: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> List[dict]:

        """
        Retrieve the most recent reports.

        Newest reports appear first.
        """

        if not self.is_connected():
            return []

        try:

            query = {}

            # Filter by prediction
            if prediction:
                query["prediction"] = prediction

            # Filter by risk level
            if risk_level:
                query["risk_level"] = risk_level

            # Fetch reports
            docs = list(
                self.collection
                .find(query)
                .sort(
                    "timestamp",
                    DESCENDING
                )
                .limit(limit)
            )

            # Convert MongoDB values
            for d in docs:

                d["_id"] = str(d["_id"])

                if "timestamp" in d:
                    d["timestamp"] = d[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            return docs

        except Exception as e:

            logger.error(
                f"Get all reports failed: {e}"
            )

            return []

    # ─────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────

    def search_reports(
        self,
        query: str,
        prediction: Optional[str] = None
    ) -> List[dict]:

        """
        Search reports by:

        - Medicine name
        - Manufacturer

        Search is case-insensitive.
        """

        if not self.is_connected():
            return []

        try:

            regex = {
                "$regex": query,
                "$options": "i"
            }

            filter_query = {
                "$or": [
                    {
                        "medicine_name": regex
                    },
                    {
                        "manufacturer": regex
                    }
                ]
            }

            # Optional prediction filter
            if prediction:
                filter_query[
                    "prediction"
                ] = prediction

            docs = list(
                self.collection
                .find(filter_query)
                .sort(
                    "timestamp",
                    DESCENDING
                )
                .limit(20)
            )

            for d in docs:

                d["_id"] = str(d["_id"])

                if "timestamp" in d:
                    d["timestamp"] = d[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            return docs

        except Exception as e:

            logger.error(
                f"Search failed: {e}"
            )

            return []

    # ─────────────────────────────────────────────
    # UPDATE
    # ─────────────────────────────────────────────

    def update_report(
        self,
        report_id: str,
        updates: dict
    ) -> bool:

        """
        Update allowed fields of an existing report.

        Used by the History page for changing:

        - Prediction
        - Risk level
        - Medicine name
        - Manufacturer
        - Batch number
        - Manufacturing date
        - Expiry date
        """

        if not self.is_connected():

            logger.warning(
                "DB not connected. Skipping update."
            )

            return False

        # Fields that are allowed to be updated
        ALLOWED_FIELDS = {
            "prediction",
            "risk_level",
            "medicine_name",
            "manufacturer",
            "batch_number",
            "manufacturing_date",
            "expiry_date",
        }

        # Keep only safe fields
        safe_updates = {
            k: v
            for k, v in updates.items()
            if k in ALLOWED_FIELDS
        }

        if not safe_updates:

            logger.warning(
                f"No allowed fields in update payload: {updates}"
            )

            return False

        try:

            from bson import ObjectId

            # Add update timestamp
            safe_updates[
                "updated_at"
            ] = datetime.utcnow()

            # Update document
            result = self.collection.update_one(
                {
                    "_id": ObjectId(report_id)
                },
                {
                    "$set": safe_updates
                }
            )

            return result.matched_count > 0

        except Exception as e:

            logger.error(
                f"Update failed: {e}"
            )

            return False

    # ─────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────

    def get_statistics(self) -> dict:

        """
        Calculate dashboard statistics.

        Returns:

        total
        genuine
        counterfeit
        average authenticity score
        """

        if not self.is_connected():

            return {
                "total": 0,
                "genuine": 0,
                "counterfeit": 0,
                "avg_score": 0,
            }

        try:

            # Total reports
            total = self.collection.count_documents({})

            # Genuine reports
            genuine = self.collection.count_documents(
                {
                    "prediction": "Genuine"
                }
            )

            # Counterfeit reports
            counterfeit = self.collection.count_documents(
                {
                    "prediction": "Counterfeit"
                }
            )

            # Average authenticity score
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg": {
                            "$avg": "$authenticity_score"
                        }
                    }
                }
            ]

            avg_result = list(
                self.collection.aggregate(
                    pipeline
                )
            )

            if (
                avg_result
                and avg_result[0].get("avg") is not None
            ):

                avg_score = round(
                    avg_result[0]["avg"],
                    1
                )

            else:

                avg_score = 0

            return {
                "total": total,
                "genuine": genuine,
                "counterfeit": counterfeit,
                "avg_score": avg_score,
            }

        except Exception as e:

            logger.error(
                f"Statistics fetch failed: {e}"
            )

            return {
                "total": 0,
                "genuine": 0,
                "counterfeit": 0,
                "avg_score": 0,
            }

    # ─────────────────────────────────────────────
    # DELETE
    # ─────────────────────────────────────────────

    def delete_report(
        self,
        report_id: str
    ) -> bool:

        """
        Delete a medicine report by ID.
        """

        if not self.is_connected():
            return False

        try:

            from bson import ObjectId

            result = self.collection.delete_one(
                {
                    "_id": ObjectId(report_id)
                }
            )

            return result.deleted_count > 0

        except Exception as e:

            logger.error(
                f"Delete failed: {e}"
            )

            return False


# ─────────────────────────────────────────────
# SINGLETON DATABASE INSTANCE
# ─────────────────────────────────────────────

db = MedSecureDB()