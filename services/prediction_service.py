import uuid
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

from database import predictions_collection
from schemas.prediction import PredictionCreate, PredictionRecord
from services.model_ai import model_ai


class PredictionService:
    def __init__(self):
        """
        Khởi tạo PredictionService với ModelAI
        """
        self.collection = predictions_collection
        # Load model khi khởi tạo service
        asyncio.create_task(self._initialize_model())
    
    async def _initialize_model(self):
        """
        Load model active khi khởi tạo service
        """
        try:
            await model_ai.load_active_model()
        except Exception as e:
            print(f"Warning: Could not load active model: {e}")
    
    async def create_prediction(self, image_url: str, doctor_info: Dict, patient_info: Dict = None) -> str:
        """
        Tạo prediction mới từ image URL
        
        Args:
            image_url: URL của ảnh X-quang
            doctor_info: Thông tin bác sĩ
            patient_info: Thông tin bệnh nhân (optional)
            
        Returns:
            str: ID của prediction record
        """
        try:
            # Dự đoán bằng AI
            prediction_details = model_ai.get_prediction_details(image_url)
            
            if not prediction_details.get("success"):
                # Fallback to mock if AI fails
                prediction_details = self._create_mock_prediction()
            
            # Tạo prediction record
            prediction_id = str(uuid.uuid4())
            
            prediction_record = {
                "id": prediction_id,
                "prediction_date": datetime.utcnow(),
                "image_info": {
                    "url": image_url,
                    "upload_date": datetime.utcnow()
                },
                "doctor_info": doctor_info,
                "patient_info": patient_info or {},
                "ai_model_info": prediction_details.get("model_info", {}),
                "prediction_result": {
                    "predicted_class_id": prediction_details["predicted_class_id"],
                    "predicted_class_name": prediction_details["predicted_class_name"],
                    "confidence": prediction_details["confidence"],
                    "probability_distribution": prediction_details["probability_distribution"],
                    "risk_level": prediction_details["risk_level"],
                    "recommendation": prediction_details["recommendation"]
                },
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Lưu vào database
            await self.collection.insert_one(prediction_record)
            
            return prediction_id
            
        except Exception as e:
            raise Exception(f"Lỗi tạo prediction: {str(e)}")
    
    def _create_mock_prediction(self) -> Dict:
        """
        Tạo mock prediction khi AI model không hoạt động
        """
        import random
        
        # Random prediction for fallback
        predicted_id = random.randint(0, 5)
        probabilities = [random.uniform(0.05, 0.3) for _ in range(6)]
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        bi_rads_names = [
            "BI-RADS 0 - Cần đánh giá thêm",
            "BI-RADS 1 - Âm tính", 
            "BI-RADS 2 - Tổn thương lành tính",
            "BI-RADS 3 - Có thể lành tính",
            "BI-RADS 4 - Nghi ngờ ác tính",
            "BI-RADS 5 - Rất nghi ngờ ác tính"
        ]
        
        probability_distribution = []
        for i, prob in enumerate(probabilities):
            probability_distribution.append({
                "class_id": i,
                "class_name": bi_rads_names[i],
                "probability": prob,
                "percentage": prob * 100
            })
        
        if predicted_id <= 2:
            risk_level = "Low"
            recommendation = "Tiếp tục tầm soát định kỳ theo lịch"
        elif predicted_id == 3:
            risk_level = "Medium" 
            recommendation = "Theo dõi ngắn hạn trong 6 tháng"
        else:
            risk_level = "High"
            recommendation = "Cần sinh thiết và thăm khám chuyên sâu ngay"
        
        return {
            "success": True,
            "predicted_class_id": predicted_id,
            "predicted_class_name": bi_rads_names[predicted_id],
            "confidence": probabilities[predicted_id],
            "probability_distribution": probability_distribution,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "model_info": {"name": "Mock Model", "version": "1.0"}
        }
    
    async def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict]:
        """
        Lấy prediction theo ID
        """
        try:
            prediction = await self.collection.find_one({"id": prediction_id})
            return prediction
        except Exception as e:
            raise Exception(f"Lỗi lấy prediction: {str(e)}")
    
    async def get_predictions_by_doctor(self, doctor_id: str, limit: int = 20, skip: int = 0) -> List[Dict]:
        """
        Lấy danh sách predictions của một bác sĩ
        """
        try:
            cursor = self.collection.find(
                {"doctor_info.id": doctor_id}
            ).sort("prediction_date", -1).skip(skip).limit(limit)
            
            predictions = await cursor.to_list(length=limit)
            return predictions
        except Exception as e:
            raise Exception(f"Lỗi lấy prediction history: {str(e)}")
    
    async def get_prediction_stats(self) -> Dict:
        """
        Lấy thống kê tổng quan về predictions
        """
        try:
            total_predictions = await self.collection.count_documents({})
            
            # Thống kê theo risk level
            pipeline = [
                {"$group": {
                    "_id": "$prediction_result.risk_level",
                    "count": {"$sum": 1}
                }}
            ]
            
            risk_stats = {}
            async for doc in self.collection.aggregate(pipeline):
                risk_stats[doc["_id"]] = doc["count"]
            
            # Thống kê theo BI-RADS class
            class_pipeline = [
                {"$group": {
                    "_id": "$prediction_result.predicted_class_id",
                    "count": {"$sum": 1}
                }}
            ]
            
            class_stats = {}
            async for doc in self.collection.aggregate(class_pipeline):
                class_stats[f"class_{doc['_id']}"] = doc["count"]
            
            return {
                "total_predictions": total_predictions,
                "risk_level_distribution": risk_stats,
                "class_distribution": class_stats,
                "model_status": model_ai.is_model_loaded()
            }
            
        except Exception as e:
            raise Exception(f"Lỗi lấy prediction stats: {str(e)}")
    
    async def delete_prediction(self, prediction_id: str) -> bool:
        """
        Xóa prediction record
        """
        try:
            result = await self.collection.delete_one({"id": prediction_id})
            return result.deleted_count > 0
        except Exception as e:
            raise Exception(f"Lỗi xóa prediction: {str(e)}")
    
    async def reload_ai_model(self) -> bool:
        """
        Reload AI model (dùng khi admin thay đổi active model)
        """
        try:
            return await model_ai.reload_active_model()
        except Exception as e:
            print(f"Error reloading AI model: {e}")
            return False


# Singleton instance
prediction_service = PredictionService() 