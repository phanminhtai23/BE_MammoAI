from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import uuid
from typing import Dict, Any, Optional, List
from utils.jwt import verify_token, verify_admin_token
from database import predictions_collection
from services.s3_client import s3_client
from collections import Counter
import calendar

router = APIRouter()

def convert_to_vietnam_time(dt: datetime) -> str:
    """Convert UTC datetime to Vietnam timezone (+7)"""
    if dt:
        vietnam_time = dt + timedelta(hours=7)
        return vietnam_time.isoformat()
    return None

@router.get('/get-all/{doctor_id}')
async def get_predictions_by_doctor_id(
    doctor_id: str,
    page: int = 1,
    limit: int = 20,
    user: dict = Depends(verify_token)
):
    try:
        skip = (page - 1) * limit
        cursor = predictions_collection.find({'doctor_id': doctor_id}).sort('created_at', -1).skip(skip).limit(limit)
        predictions = await cursor.to_list(length=limit)
        total = await predictions_collection.count_documents({'doctor_id': doctor_id})

        result = []
        for prediction in predictions:
            result.append({
                'id': prediction.get('id'),
                'doctor_id': prediction.get('doctor_id'),
                'image_url': prediction.get('image_url'),
                'image_original_name': prediction.get('image_original_name'),
                'image_key': prediction.get('image_key'),
                'model_name': prediction.get('model_name'),
                'prediction_result': prediction.get('prediction_result'),
                'probability': prediction.get('probability'),
                'created_at': convert_to_vietnam_time(prediction.get('created_at')),
                'updated_at': convert_to_vietnam_time(prediction.get('updated_at'))
            })

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': result,
                'total': total,
                'page': page,
                'limit': limit,
                'message': f'Lấy danh sách predictions của doctor {doctor_id} thành công'
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy danh sách predictions theo doctor_id: {str(e)}'
        )

@router.get('/get-all')
async def get_all_predictions(
    page: int = 1,
    limit: int = 20,
    search: Optional[str] = None,
    model_filter: Optional[str] = None,
    result_filter: Optional[str] = None,
    user: dict = Depends(verify_admin_token)
):
    """
    Lấy toàn bộ danh sách predictions (admin) với filter
    """
    try:
        # Tạo filter query
        filter_query = {}
        
        if search:
            filter_query['image_original_name'] = {'$regex': search, '$options': 'i'}
        
        if model_filter:
            filter_query['model_name'] = model_filter
            
        if result_filter:
            filter_query['prediction_result'] = result_filter

        skip = (page - 1) * limit
        predictions_cursor = predictions_collection.find(filter_query).sort('created_at', -1).skip(skip).limit(limit)
        predictions = await predictions_cursor.to_list(length=limit)
        total = await predictions_collection.count_documents(filter_query)

        result = []
        for prediction in predictions:
            result.append({
                'id': prediction.get('id'),
                'doctor_id': prediction.get('doctor_id'),
                'image_url': prediction.get('image_url'),
                'image_original_name': prediction.get('image_original_name'),
                'image_key': prediction.get('image_key'),
                'model_name': prediction.get('model_name'),
                'prediction_result': prediction.get('prediction_result'),
                'probability': prediction.get('probability'),
                'created_at': convert_to_vietnam_time(prediction.get('created_at')),
                'updated_at': convert_to_vietnam_time(prediction.get('updated_at'))
            })

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': result,
                'total': total,
                'page': page,
                'limit': limit,
                'message': 'Lấy danh sách predictions thành công'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy danh sách predictions: {str(e)}'
        )

@router.get('/{prediction_id}')
async def get_prediction_by_id(prediction_id: str, user: dict = Depends(verify_token)):
    """
    Lấy prediction theo ID
    """
    try:
        # Tìm prediction trong MongoDB
        prediction = await predictions_collection.find_one({'id': prediction_id})
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail='Không tìm thấy prediction với ID này'
            )
        
        # Chuyển đổi kết quả thành dictionary
        result = {
            'id': prediction.get('id'),
            'doctor_id': prediction.get('doctor_id'),
            'image_url': prediction.get('image_url'),
            'image_original_name': prediction.get('image_original_name'),
            'image_key': prediction.get('image_key'),
            'model_name': prediction.get('model_name'),
            'prediction_result': prediction.get('prediction_result'),
            'probability': prediction.get('probability'),
            'created_at': convert_to_vietnam_time(prediction.get('created_at')),
            'updated_at': convert_to_vietnam_time(prediction.get('updated_at'))
        }
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': result,
                'message': 'Lấy prediction thành công'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy prediction: {str(e)}'
        )

@router.delete('/{prediction_id}')
async def delete_prediction(prediction_id: str, file_key: str, user: dict = Depends(verify_token)):
    """
    Xóa prediction theo ID
    """
    try:
        # Kiểm tra xem prediction có tồn tại không
        existing_prediction = await predictions_collection.find_one({'id': prediction_id})
        
        if not existing_prediction:
            raise HTTPException(
                status_code=404,
                detail='Không tìm thấy prediction với ID này'
            )
        
        # Xóa prediction
        await predictions_collection.delete_one({'id': prediction_id})
        print("xóa file key", file_key)
        await s3_client.delete_image(file_key)

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'message': 'Xóa prediction thành công'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi xóa prediction: {str(e)}'
        )

@router.get('/statistics/daily')
async def get_daily_prediction_stats(
    days: int = 30,
    doctor_id: Optional[str] = None,
    user: dict = Depends(verify_token)
):
    """
    Lấy thống kê predictions theo ngày để vẽ biểu đồ
    """
    try:
        # Tính ngày bắt đầu theo múi giờ Việt Nam (+7)
        vietnam_now = datetime.utcnow() + timedelta(hours=7)
        end_date_vietnam = vietnam_now.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date_vietnam = (vietnam_now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Chuyển về UTC để query database
        end_date_utc = end_date_vietnam - timedelta(hours=7)
        start_date_utc = start_date_vietnam - timedelta(hours=7)
        
        # print(f"🔍 Daily Stats - Vietnam now: {vietnam_now.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 Daily Stats - Start date (Vietnam): {start_date_vietnam.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 Daily Stats - End date (Vietnam): {end_date_vietnam.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 Daily Stats - Start date (UTC): {start_date_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 Daily Stats - End date (UTC): {end_date_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Tạo filter query
        filter_query = {
            'created_at': {
                '$gte': start_date_utc,
                '$lte': end_date_utc
            }
        }
        
        if doctor_id:
            filter_query['doctor_id'] = doctor_id

        # Lấy predictions trong khoảng thời gian
        predictions = await predictions_collection.find(filter_query).to_list(length=None)
        print(f"🔍 Daily Stats - Found {len(predictions)} predictions")
        
        # Tạo dictionary để đếm theo ngày (Vietnam time)
        daily_counts = {}
        
        # Khởi tạo tất cả các ngày trong khoảng với count = 0 (Vietnam time)
        current_date = start_date_vietnam
        while current_date <= end_date_vietnam:
            date_key = current_date.strftime('%Y-%m-%d')
            daily_counts[date_key] = 0
            current_date += timedelta(days=1)
        
        # Đếm predictions theo ngày
        for pred in predictions:
            created_at = pred.get('created_at')
            if created_at:
                # Chuyển về Vietnam time (+7)
                vietnam_time = created_at + timedelta(hours=7)
                date_key = vietnam_time.strftime('%Y-%m-%d')
                if date_key in daily_counts:
                    daily_counts[date_key] += 1
                # print(f"🔍 Daily Stats - Prediction: {created_at} -> Vietnam: {vietnam_time} -> Date: {date_key}")
        
        # Chuyển thành array format cho chart
        daily_stats = [
            {'date': date, 'count': count} 
            for date, count in sorted(daily_counts.items())
        ]

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': daily_stats,
                'message': 'Lấy thống kê daily predictions thành công'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy thống kê daily predictions: {str(e)}'
        )

@router.get('/statistics/class-distribution')
async def get_class_distribution_stats(
    doctor_id: Optional[str] = None,
    user: dict = Depends(verify_token)
):
    """
    Lấy thống kê phân bố class để vẽ biểu đồ cột
    """
    try:
        # Tạo filter query
        filter_query = {}
        
        if doctor_id:
            filter_query['doctor_id'] = doctor_id

        # Lấy tất cả predictions
        predictions = await predictions_collection.find(filter_query).to_list(length=None)
        
        # Đếm theo prediction_result
        class_counts = Counter()
        total_images = 0
        
        for pred in predictions:
            result = pred.get('prediction_result')
            if result:
                class_counts[result] += 1
                total_images += 1
        
        # Chuyển thành format cho chart
        class_stats = dict(class_counts)

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'class_stats': class_stats,
                    'total_images': total_images
                },
                'message': 'Lấy thống kê class distribution thành công'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy thống kê class distribution: {str(e)}'
        )

@router.get('/statistics/admin-daily')
async def get_admin_daily_stats(
    days: int = 30,
    user: dict = Depends(verify_admin_token)
):
    """
    Lấy thống kê daily cho admin (tất cả doctors)
    """
    try:
        # Tính ngày bắt đầu
        # Lấy thời gian hiện tại theo múi giờ Việt Nam (+7)
        vietnam_now = datetime.utcnow() + timedelta(hours=7)
        end_date = vietnam_now.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = (vietnam_now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Chuyển về UTC để query database
        end_date_utc = end_date - timedelta(hours=7)
        start_date_utc = start_date - timedelta(hours=7)
        
        # print(f"🔍 Vietnam time: {vietnam_now.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 Start date (Vietnam): {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 End date (Vietnam): {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 Start date (UTC): {start_date_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"🔍 End date (UTC): {end_date_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Tạo filter query
        filter_query = {
            'created_at': {
                '$gte': start_date_utc,
                '$lte': end_date_utc
            }
        }

        # Lấy predictions trong khoảng thời gian
        predictions = await predictions_collection.find(filter_query).to_list(length=None)
        
        # Tạo dictionary để đếm theo ngày
        daily_counts = {}
        
        # Khởi tạo tất cả các ngày trong khoảng với count = 0
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            daily_counts[date_key] = 0
            current_date += timedelta(days=1)
        
        # Đếm predictions theo ngày
        for pred in predictions:
            created_at = pred.get('created_at')
            if created_at:
                # Chuyển về Vietnam time
                vietnam_time = created_at + timedelta(hours=8)
                date_key = vietnam_time.strftime('%Y-%m-%d')
                if date_key in daily_counts:
                    daily_counts[date_key] += 1
        
        # Chuyển thành array format cho chart
        daily_stats = [
            {'date': date, 'count': count} 
            for date, count in sorted(daily_counts.items())
        ]

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': daily_stats,
                'message': 'Lấy thống kê admin daily predictions thành công'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy thống kê admin daily predictions: {str(e)}'
        )

@router.get('/statistics/admin-class-distribution')
async def get_admin_class_distribution_stats(
    user: dict = Depends(verify_admin_token)
):
    """
    Lấy thống kê phân bố class cho admin (tất cả doctors)
    """
    try:
        # Lấy tất cả predictions
        predictions = await predictions_collection.find({}).to_list(length=None)
        
        # Đếm theo prediction_result
        class_counts = Counter()
        total_images = 0
        
        for pred in predictions:
            result = pred.get('prediction_result')
            if result:
                class_counts[result] += 1
                total_images += 1
        
        # Chuyển thành format cho chart
        class_stats = dict(class_counts)

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'class_stats': class_stats,
                    'total_images': total_images
                },
                'message': 'Lấy thống kê admin class distribution thành công'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy thống kê admin class distribution: {str(e)}'
        )

@router.get('/statistics/admin-average-confidence')
async def get_admin_average_confidence(
    user: dict = Depends(verify_admin_token)
):
    """
    Lấy độ tin cậy trung bình cho admin (tất cả doctors)
    """
    try:
        # Lấy tất cả predictions
        predictions = await predictions_collection.find({}).to_list(length=None)
        
        # Tính toán độ tin cậy trung bình
        total_probability = 0
        valid_predictions = 0
        
        for pred in predictions:
            probability = pred.get('probability')
            if probability is not None:
                total_probability += probability
                valid_predictions += 1
        
        average_confidence = 0
        if valid_predictions > 0:
            average_confidence = round(total_probability / valid_predictions, 2)

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'average_confidence': average_confidence,
                    'total_predictions': valid_predictions
                },
                'message': 'Lấy độ tin cậy trung bình admin thành công'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Lỗi khi lấy độ tin cậy trung bình admin: {str(e)}'
        )

