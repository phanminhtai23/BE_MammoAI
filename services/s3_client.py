import boto3
import os
from typing import List, Dict, Optional

# from botocore.exceptions import ClientError
import logging
import uuid
from datetime import datetime


class S3Client:
    def __init__(self):
        """Initialize S3 client with configuration from environment variables"""
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME")

        # Folder configurations
        self.images_folder = "mammo-images"
        self.models_folder = "mammo-models"

    def _generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate unique filename similar to frontend implementation

        Args:
            original_filename: Original filename with extension

        Returns:
            Unique filename with timestamp and UUID
        """
        # Get file extension
        file_extension = original_filename.split(".")[-1].lower()

        # Generate timestamp in ISO format, replace colons and dots with dashes
        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")

        # Generate unique ID (first 8 characters of UUID)
        unique_id = str(uuid.uuid4())[:8]

        # Create unique filename
        unique_filename = f"{timestamp}_{unique_id}.{file_extension}"

        return unique_filename

    def list_images(self) -> List[Dict]:
        """
        List all images in mammo-images folder

        Returns:
            List of image objects with metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=f"{self.images_folder}/"
            )

            images = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    if obj["Key"] != f"{self.images_folder}/":  # Skip folder itself
                        image_info = {
                            "key": obj["Key"],
                            "filename": obj["Key"].split("/")[-1],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "url": f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}",
                        }
                        images.append(image_info)

            return images

        except Exception as e:
            return []

    def list_models(self) -> List[Dict]:
        """
        List all models in mammoai-models folder

        Returns:
            List of model objects with metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=f"{self.models_folder}/"
            )

            models = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    if obj["Key"] != f"{self.models_folder}/":  # Skip folder itself
                        model_info = {
                            "key": obj["Key"],
                            "filename": obj["Key"].split("/")[-1],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "url": f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}",
                        }
                        models.append(model_info)

            return models

        except Exception as e:
            return []

    async def delete_image(self, file_key: str) -> Dict:
        """
        Delete specific image from S3

        Args:
            file_key: S3 key of the image to delete

        Returns:
            Dict containing deletion result
        """
        try:
            # Ensure the key is in the images folder
            if not file_key.startswith(f"{self.images_folder}/"):
                file_key = f"{self.images_folder}/{file_key}"

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_key)

            return {"success": True, "deleted_key": file_key}

        except Exception as e:
            raise e

    async def delete_model(self, file_key: str) -> Dict:
        """
        Delete specific model from S3

        Args:
            file_key: S3 key of the model to delete

        Returns:
            Dict containing deletion result
        """
        try:
            # Ensure the key is in the models folder
            if not file_key.startswith(f"{self.models_folder}/"):
                file_key = f"{self.models_folder}/{file_key}"

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_key)

            print(f"✅ Model deleted successfully: {file_key}")
            return {"success": True, "deleted_key": file_key}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_all_images(self) -> Dict:
        """
        Delete all images in mammo-images folder

        Returns:
            Dict containing deletion result
        """
        try:
            images = self.list_images()
            deleted_count = 0

            for image in images:
                result = self.delete_image(image["key"])
                if result["success"]:
                    deleted_count += 1

            return {
                "success": True,
                "deleted_count": deleted_count,
                "total_count": len(images),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_all_models(self) -> Dict:
        """
        Delete all models in mammoai-models folder

        Returns:
            Dict containing deletion result
        """
        try:
            models = self.list_models()
            deleted_count = 0

            for model in models:
                result = self.delete_model(model["key"])
                if result["success"]:
                    deleted_count += 1

            return {
                "success": True,
                "deleted_count": deleted_count,
                "total_count": len(models),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # def upload_image(self, file_path: str, file_name: str = None) -> Dict:
    #     """
    #     Upload image to S3 mammo-images folder with unique filename

    #     Args:
    #         file_path: Local path to the file
    #         file_name: Original filename (optional, will use basename if not provided)

    #     Returns:
    #         Dict containing upload result
    #     """
    #     try:
    #         # Generate unique filename if not provided
    #         if file_name is None:
    #             file_name = os.path.basename(file_path)

    #         unique_filename = self._generate_unique_filename(file_name)
    #         s3_key = f"{self.images_folder}/{unique_filename}"

    #         self.s3_client.upload_file(
    #             file_path,
    #             self.bucket_name,
    #             s3_key,
    #             ExtraArgs={
    #                 'ACL': 'public-read',
    #                 'ContentType': self._get_content_type(unique_filename),
    #                 'Metadata': {
    #                     'original-filename': file_name,
    #                     'upload-time': datetime.now().isoformat(),
    #                     'file-size': str(os.path.getsize(file_path))
    #                 }
    #             }
    #         )

    #         url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"

    #         logger.info(f"✅ Image uploaded successfully: {url}")
    #         return {
    #             'success': True,
    #             'url': url,
    #             'key': s3_key,
    #             'filename': unique_filename,
    #             'original_filename': file_name
    #         }

    #     except Exception as e:
    #         logger.error(f"❌ Error uploading image: {str(e)}")
    #         return {
    #             'success': False,
    #             'error': str(e)
    #         }

    # def upload_model(self, file_path: str, file_name: str = None) -> Dict:
    #     """
    #     Upload model to S3 mammo-models folder with unique filename

    #     Args:
    #         file_path: Local path to the file
    #         file_name: Original filename (optional, will use basename if not provided)

    #     Returns:
    #         Dict containing upload result
    #     """
    #     try:
    #         # Generate unique filename if not provided
    #         if file_name is None:
    #             file_name = os.path.basename(file_path)

    #         unique_filename = self._generate_unique_filename(file_name)
    #         s3_key = f"{self.models_folder}/{unique_filename}"

    #         self.s3_client.upload_file(
    #             file_path,
    #             self.bucket_name,
    #             s3_key,
    #             ExtraArgs={
    #                 'ACL': 'public-read',
    #                 'ContentType': self._get_content_type(unique_filename),
    #                 'Metadata': {
    #                     'original-filename': file_name,
    #                     'upload-time': datetime.now().isoformat(),
    #                     'file-size': str(os.path.getsize(file_path))
    #                 }
    #             }
    #         )

    #         url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"

    #         logger.info(f"✅ Model uploaded successfully: {url}")
    #         return {
    #             'success': True,
    #             'url': url,
    #             'key': s3_key,
    #             'filename': unique_filename,
    #             'original_filename': file_name
    #         }

    #     except Exception as e:
    #         logger.error(f"❌ Error uploading model: {str(e)}")
    #         return {
    #             'success': False,
    #             'error': str(e)
    #         }


# Create a singleton instance
s3_client = S3Client()
