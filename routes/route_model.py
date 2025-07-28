from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
import uuid
from datetime import datetime, timezone, timedelta

from services.s3_client import s3_client
from database import models_collection, predictions_collection
from utils.jwt import verify_admin_token, verify_token
from schemas.model import ModelInfor, ModelCreate, ModelUpdate
from services.model_ai import model_ai

router = APIRouter()

from pydantic import BaseModel


class PredictRequest(BaseModel):
    doctor_id: str
    image_url: str
    image_original_name: str
    image_key: str
    model_name: str


class PredictionSchema(BaseModel):
    id: str
    doctor_id: str
    image_url: str
    image_original_name: str
    image_key: str
    model_name: str
    prediction_result: str
    probability: float
    created_at: datetime
    updated_at: datetime


async def swap_model(model_id: str):
    # Check if model exists
    existing_model = await models_collection.find_one({"id": model_id})
    if not existing_model:
        raise HTTPException(status_code=404, detail="Không tìm thấy model")

    await model_ai.reload_model(existing_model["model_url"])
    # set all other models to inactive
    await models_collection.update_many(
        {"is_active": True}, {"$set": {"is_active": False}}
    )
    # Update is_active to True
    result = await models_collection.update_one(
        {"id": model_id}, {"$set": {"is_active": True}}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Model đã được kích hoạt rồi")

    return {
        "success": True,
        "message": "Model đã được kích hoạt thành công",
        "model_id": model_id,
    }


@router.post("/predict")
async def predict(request: PredictRequest, user=Depends(verify_token)):
    try:

        # print("image_url: \n", request)
        prediction_result = await model_ai.predict(request.image_url)

        # print("prediction_result: \n", prediction_result)

        # Tìm lớp có xác suất cao nhất
        max_probability = max(prediction_result)
        max_class_index = prediction_result.index(max_probability)

        # Mapping tên lớp BI-RADS
        bi_rads_names = {
            0: "BI-RADS 0",
            1: "BI-RADS 1",
            2: "BI-RADS 2",
            3: "BI-RADS 3",
            4: "BI-RADS 4",
            5: "BI-RADS 5",
        }

        predicted_class_name = bi_rads_names.get(
            max_class_index, f"Class {max_class_index}"
        )

        print(f"🎯 Predicted class: {max_class_index} ({predicted_class_name})")
        print(f"🎯 Confidence: {max_probability:.4f} ({max_probability*100:.2f}%)")

        # Save prediction result to database với thông tin chi tiết
        prediction_record = PredictionSchema(
            id=str(uuid.uuid4()),
            doctor_id=request.doctor_id,
            image_url=request.image_url,
            image_original_name=request.image_original_name,
            image_key=request.image_key,
            model_name=request.model_name,
            prediction_result=predicted_class_name,  # Lưu thông tin chi tiết thay vì chỉ probabilities
            probability=round(max_probability * 100, 2),
            created_at=datetime.now(timezone(timedelta(hours=8))),
            updated_at=datetime.now(timezone(timedelta(hours=8))),
        )
        await predictions_collection.insert_one(prediction_record.model_dump())

        return prediction_result  # Trả về thông tin chi tiết
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")


@router.get("/infor-model")
async def get_infor_model(user=Depends(verify_token)):
    """
    Lấy thông tin model đang active (name và version)
    """
    try:
        # Tìm model có is_active = True
        active_model = await models_collection.find_one({"is_active": True})

        if not active_model:
            raise HTTPException(
                status_code=404, detail="Không có model nào đang active"
            )

        return {
            "name": active_model.get("name"),
            "version": active_model.get("version"),
            "model_url": active_model.get("model_url"),
            "model_key": active_model.get("model_key"),
            "model_original_name": active_model.get("model_original_name"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi lấy thông tin model: {str(e)}"
        )


@router.get("/model-is-availabe")
async def model_is_availabe(user=Depends(verify_token)):
    """
    Kiểm tra model có sẵn trong database không
    """
    print("ahiaiah", model_ai.current_model is not None)
    try:
        result = await model_ai.initialize_model()
        return {"available": result}
    except Exception as e:
        print(f"❌ Lỗi kiểm tra model: {e}")
        return {"available": False}


@router.post("/", response_model=ModelInfor)
async def create_model(
    model_data: ModelCreate, admin_user: dict = Depends(verify_admin_token)
):
    """
    Tạo model mới
    """
    try:
        print("model_data: ", model_data)
        # print("admin_user: ", admin_user)
        # Check if model name already exists
        existing_model = await models_collection.find_one(
            {"name": model_data.name, "version": model_data.version}
        )
        if existing_model:
            raise HTTPException(
                status_code=400, detail="Model với tên và phiên bản này đã tồn tại"
            )

        # Generate model ID
        model_id = str(uuid.uuid4())
        current_time = datetime.now(timezone(timedelta(hours=7)))

        if model_data.is_active is True:
            # Set all other models to inactive
            await models_collection.update_many(
                {"is_active": True}, {"$set": {"is_active": False}}
            )

            # Create model record
            model_record = ModelInfor(
                id=model_id,
                name=model_data.name,
                version=model_data.version,
                model_url=model_data.model_url,
                model_key=model_data.model_key,
                model_original_name=model_data.model_original_name,
                accuracy=model_data.accuracy,
                is_active=model_data.is_active,
                created_at=current_time,
                updated_at=current_time,
            )

            print("đã thêm và set model mới nhất lên")
            # set model mới nhất lên

            result = await models_collection.insert_one(model_record.model_dump())
            await swap_model(model_id)

            if result:
                print("đã thêm và set model mới nhất lên")
                # set model mới nhất lên

        else:

            # chỉ thêm và lưu
            model_record = ModelInfor(
                id=model_id,
                name=model_data.name,
                model_url=model_data.model_url,
                model_key=model_data.model_key,
                model_original_name=model_data.model_original_name,
                version=model_data.version,
                accuracy=model_data.accuracy,
                is_active=model_data.is_active,
                created_at=current_time,
                updated_at=current_time,
            )
            await models_collection.insert_one(model_record.model_dump())
            print("đã thêm model")
        return model_record

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo model: {str(e)}")


# lấy danh sách model
@router.get("/", response_model=List[ModelInfor])
async def get_models(
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    admin_user: dict = Depends(verify_admin_token),
):
    """
    Lấy danh sách tất cả models với filtering và pagination
    """
    try:
        # Validate pagination
        if page < 1 or limit < 1 or limit > 100:
            raise HTTPException(
                status_code=400, detail="Tham số phân trang không hợp lệ"
            )

        # Build filter query
        filter_query = {}
        if is_active is not None:
            filter_query["is_active"] = is_active
        if search:
            filter_query["$or"] = [
                {"name": {"$regex": search, "$options": "i"}},
                {"version": {"$regex": search, "$options": "i"}},
            ]

        # Calculate skip
        skip = (page - 1) * limit

        # Get models from database
        cursor = (
            models_collection.find(filter_query).skip(skip).limit(limit).sort("name", 1)
        )
        models = await cursor.to_list(length=limit)

        # Convert to response format with timezone conversion
        model_responses = []
        for model in models:
            # Chuyển đổi múi giờ từ UTC+0 sang UTC+7 cho created_at và updated_at
            if "created_at" in model and model["created_at"]:
                model["created_at"] = model["created_at"] + timedelta(hours=7)
            if "updated_at" in model and model["updated_at"]:
                model["updated_at"] = model["updated_at"] + timedelta(hours=7)

            model_responses.append(ModelInfor(**model))

        return model_responses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi lấy danh sách model: {str(e)}"
        )


@router.get("/{model_id}", response_model=ModelInfor)
async def get_model(model_id: str, admin_user: dict = Depends(verify_admin_token)):
    """
    Lấy thông tin chi tiết một model theo ID
    """
    try:
        # Find model by ID
        model = await models_collection.find_one({"id": model_id})
        if not model:
            raise HTTPException(status_code=404, detail="Không tìm thấy model")

        return ModelInfor(**model)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi lấy thông tin model: {str(e)}"
        )


@router.put("/{model_id}", response_model=ModelInfor)
async def update_model(
    model_id: str,
    update_data: ModelUpdate,
    admin_user: dict = Depends(verify_admin_token),
):
    """
    Cập nhật thông tin model
    """
    try:
        # Check if model exists
        existing_model = await models_collection.find_one({"id": model_id})
        if not existing_model:
            raise HTTPException(status_code=404, detail="Không tìm thấy model")

        # Prepare update data
        update_dict = {}
        for field, value in update_data.dict(exclude_unset=True).items():
            if value is not None:
                update_dict[field] = value

        if not update_dict:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")

        # Update model
        # Thêm updated_at với múi giờ +7
        update_dict["updated_at"] = datetime.now(timezone(timedelta(hours=7)))

        result = await models_collection.update_one(
            {"id": model_id}, {"$set": update_dict}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=400, detail="Không có thay đổi nào được thực hiện"
            )

        # Get updated model
        updated_model = await models_collection.find_one({"id": model_id})
        return ModelInfor(**updated_model)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cập nhật model: {str(e)}")


@router.delete("/{model_id}")
async def delete_model(model_id: str, admin_user: dict = Depends(verify_admin_token)):
    """
    Xóa model
    """
    try:
        # Check if model exists
        existing_model = await models_collection.find_one({"id": model_id})
        if not existing_model:
            raise HTTPException(status_code=404, detail="Không tìm thấy model")
        await s3_client.delete_model(existing_model["model_key"])
        # Delete model from database
        result = await models_collection.delete_one({"id": model_id})

        if result.deleted_count == 0:
            raise HTTPException(status_code=500, detail="Không thể xóa model")

        return {
            "success": True,
            "message": "Model đã được xóa thành công",
            "deleted_model_id": model_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa model: {str(e)}")


@router.patch("/{model_id}/activate")
async def activate_model(model_id: str, admin_user: dict = Depends(verify_admin_token)):
    """
    Kích hoạt model
    """
    try:
        await swap_model(model_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi kích hoạt model: {str(e)}")


# End of routes
