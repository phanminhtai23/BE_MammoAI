import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import asyncio
import aiohttp
import gc  # Garbage collection để giải phóng RAM
from database import models_collection


class ModelAI:
    def __init__(self):
        """
        Khởi tạo ModelAI class để load model từ local file
        """
        self.current_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ["0", "1", "2", "3", "4", "5"]

        # Model directory và file path
        self.model_dir = Path("./ai_model")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "best_checkpoint.pt"

        # Transform cho ảnh (ImageNet pretrained)
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.size = (299, 299)

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ]
        )

        print(f"🧠 ModelAI initialized")
        print(f"📂 Model path: {self.model_path}")
        print(f"🎯 Device: {self.device}")

        # Không tự động load model trong __init__, sẽ load khi cần
        print("💡 Model sẽ được load khi cần thiết")

    def _clear_model_from_memory(self):
        """
        Xóa model khỏi RAM và giải phóng memory
        """
        if self.current_model is not None:
            # Move model to CPU trước khi xóa (giải phóng VRAM nếu dùng GPU)
            if hasattr(self.current_model, "cpu"):
                self.current_model.cpu()

            # Xóa reference đến model
            del self.current_model
            self.current_model = None

            # Clear CUDA cache nếu dùng GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection để giải phóng RAM
            gc.collect()

            print("🗑️ Đã xóa model khỏi RAM")

    def _create_resnet_with_dropout(
        self, num_classes=6, is_gray=False, dropout_prob=0.4
    ):
        """
        Tạo custom ResNet với Dropout như của anh
        """

        class ResNetWithDropout(nn.Module):
            def __init__(self, num_classes=6, is_gray=False, dropout_prob=0.4):
                super().__init__()
                self.is_gray = is_gray
                from torchvision import models

                self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                if is_gray:
                    self.resnet.conv1 = nn.Conv2d(
                        1, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                self.dropout = nn.Dropout(p=dropout_prob)
                self.fc = nn.Linear(512 * 4, num_classes)

            def forward(self, x):
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)
                x = self.resnet.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.dropout(x)
                return self.fc(x)

        print("🎯 Creating custom ResNetWithDropout model...")
        model = ResNetWithDropout(
            num_classes=num_classes, is_gray=is_gray, dropout_prob=dropout_prob
        )
        print("✅ Custom ResNetWithDropout created successfully")
        return model

    def _create_inception_v3_with_dropout(
        self, num_classes=6, is_gray=False, dropout_prob=0.4
    ):
        """
        Tạo custom InceptionV3 với Dropout
        """

        class InceptionV3WithDropout(nn.Module):
            def __init__(self, num_classes=6, is_gray=False, dropout_prob=0.2):
                super().__init__()
                self.is_gray = is_gray
                from torchvision import models

                # Load pretrained InceptionV3 with auxiliary logits enabled for training
                # Need specify weights=... and aux_logits=True for training
                self.model = models.inception_v3(
                    weights=models.Inception_V3_Weights.DEFAULT,
                    aux_logits=True,
                    transform_input=False,
                )

                if self.is_gray:
                    # InceptionV3 sử dụng Conv2d_1a_3x3 thay vì conv1
                    original_first_layer = self.model.Conv2d_1a_3x3.conv
                    self.model.Conv2d_1a_3x3.conv = nn.Conv2d(
                        1,  # Từ 3 channels xuống 1 channel
                        original_first_layer.out_channels,
                        kernel_size=original_first_layer.kernel_size,
                        stride=original_first_layer.stride,
                        padding=original_first_layer.padding,
                        bias=False,
                    )

                # Replace main classifier with dropout + linear
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_prob), nn.Linear(num_ftrs, num_classes)
                )

                # Replace the auxiliary classifier (AuxLogits.fc)
                # Check if aux_logits is enabled
                if self.model.AuxLogits is not None:
                    num_aux_ftrs = self.model.AuxLogits.fc.in_features
                    self.model.AuxLogits.fc = nn.Sequential(
                        nn.Dropout(p=dropout_prob), nn.Linear(num_aux_ftrs, num_classes)
                    )
                else:
                    print(
                        "Warning: AuxLogits is None, auxiliary classifier not replaced."
                    )

            def forward(self, x):
                # InceptionV3 returns a tuple (output, aux_output) during training
                # and just output during evaluation. The train_model function handles this.
                return self.model(x)

        print("🎯 Creating custom InceptionV3WithDropout model...")
        model = InceptionV3WithDropout(
            num_classes=num_classes, is_gray=is_gray, dropout_prob=dropout_prob
        )
        print("✅ Custom InceptionV3WithDropout created successfully")
        return model

    def _create_mobilenet_v2_with_dropout(
        self, num_classes=6, is_gray=False, dropout_prob=0.2
    ):
        """
        Tạo custom MobileNetV2 với Dropout
        """

        class MobileNetV2WithDropout(nn.Module):
            def __init__(self, num_classes=6, is_gray=False, dropout_prob=0.2):
                super().__init__()
                self.is_gray = is_gray
                from torchvision import models

                # Load pretrained MobileNetV2
                self.model = models.mobilenet_v2(
                    weights=models.MobileNet_V2_Weights.DEFAULT
                )

                if self.is_gray:
                    # Thay đổi input layer từ 3 channels thành 1 channel
                    original_first_layer = self.model.features[0][0]
                    self.model.features[0][0] = nn.Conv2d(
                        1,  # Từ 3 channels xuống 1 channel
                        original_first_layer.out_channels,
                        kernel_size=original_first_layer.kernel_size,
                        stride=original_first_layer.stride,
                        padding=original_first_layer.padding,
                        bias=False,
                    )

                # Replace classifier với dropout + linear
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_prob), nn.Linear(num_ftrs, num_classes)
                )

            def forward(self, x):
                return self.model(x)

        print("🎯 Creating custom MobileNetV2WithDropout model...")
        model = MobileNetV2WithDropout(
            num_classes=num_classes, is_gray=is_gray, dropout_prob=dropout_prob
        )
        print("✅ Custom MobileNetV2WithDropout created successfully")
        return model

    async def load_model(self, current_model: dict) -> bool:
        """
        Load model từ file local backend/ai_model/best_checkpoint.pt

        Returns:
            bool: True nếu load thành công
        """
        try:
            if not self.model_path.exists():
                print(f"⚠️ Model file không tồn tại: {self.model_path}")
                print(f"Tiến hành tải model từ {current_model['model_url']}")
                result_download = await self._download_model(
                    current_model["model_url"], self.model_path
                )

                if result_download:
                    print(f"✅ Tải model thành công từ {current_model['model_url']}")

            print(f"🔄 Loading model từ: {self.model_path}")

            # Xóa model cũ khỏi RAM trước
            self._clear_model_from_memory()

            model = None

            if current_model["name"] == "ResNet50":
                # Tạo custom ResNetWithDropout model
                model = self._create_resnet_with_dropout(
                    num_classes=6,  # BI-RADS 0-5
                    is_gray=False,  # RGB images (anh có thể thay đổi thành True nếu dùng grayscale)
                    dropout_prob=0.4,  # Dropout rate
                )
                # Load PyTorch checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint["model_state_dict"]

                # Handle DataParallel prefix
                if list(state_dict.keys())[0].startswith("module."):
                    from collections import OrderedDict

                    state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

                # Load state dict vào model
                model.load_state_dict(state_dict)

            elif current_model["name"] == "InceptionV3":
                model = self._create_inception_v3_with_dropout(
                    num_classes=6, is_gray=False, dropout_prob=0.2
                )
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint["model_state_dict"]
                if list(state_dict.keys())[0].startswith("module."):
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)

            elif current_model["name"] == "MobileNetV2":
                model = self._create_mobilenet_v2_with_dropout(
                    num_classes=6, is_gray=False, dropout_prob=0.2
                )
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Xử lý trường hợp model được train bằng DataParallel
                state_dict = checkpoint["model_state_dict"]
                # Kiểm tra nếu state_dict có tiền tố 'module.' (từ DataParallel)
                if list(state_dict.keys())[0].startswith("module."):
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)

            else:
                raise ValueError(f"Model {current_model['name']} không hợp lệ")

            # Move model to device và set eval mode
            model = model.to(self.device)
            model.eval()

            # Clear checkpoint từ memory để tiết kiệm RAM
            del checkpoint
            del state_dict
            gc.collect()

            self.current_model = model
            print(f"✅ Model {current_model['name']} loaded thành công!")
            return True

        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            self._clear_model_from_memory()
            return False

    async def reload_model(self, model_url: str) -> bool:
        """
        Download model từ URL, thay thế model cũ và load model mới

        Args:
            model_url: URL của model cần download

        Returns:
            bool: True nếu reload thành công
        """
        try:
            print(f"🔄 Reloading model từ URL: {model_url}")

            # Xóa model cũ khỏi RAM trước
            self._clear_model_from_memory()

            # Xóa file model cũ nếu tồn tại
            if self.model_path.exists():
                self.model_path.unlink()
                print(f"🗑️ Đã xóa model cũ: {self.model_path}")

            # Download model mới từ URL
            print(f"📥 Downloading model từ URL...")
            success = await self._download_model(model_url, self.model_path)

            if not success:
                print("❌ Download model thất bại")
                return False

            # Load model mới
            print(f"🧠 Loading model mới...")
            current_model = await self.get_model_info()
            if current_model:
                load_success = await self.load_model(current_model)
            else:
                print("❌ Không tìm thấy model active để load")
                return False

            if load_success:
                print(f"✅ Reload model thành công!")
                return True
            else:
                print("❌ Load model mới thất bại")
                return False

        except Exception as e:
            print(f"❌ Lỗi reload model: {e}")
            self._clear_model_from_memory()
            return False

    async def _download_model(self, model_url: str, save_path: Path) -> bool:
        """
        Download model từ URL với retry và timeout
        """
        max_retries = 3
        timeout = 300  # 5 minutes timeout

        for attempt in range(max_retries):
            try:
                connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
                timeout_config = aiohttp.ClientTimeout(total=timeout)

                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout_config
                ) as session:
                    async with session.get(model_url) as response:
                        if response.status == 200:
                            with open(save_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)

                            # Check file size
                            file_size = save_path.stat().st_size / (1024 * 1024)  # MB
                            print(
                                f"✅ Downloaded: {save_path.name} ({file_size:.1f} MB)"
                            )
                            return True
                        else:
                            print(f"❌ HTTP {response.status}")

            except asyncio.TimeoutError:
                print(f"⏰ Timeout attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                print(f"❌ Download error attempt {attempt + 1}/{max_retries}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return False

    async def predict_image(
        self, image_url: str
    ) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[List[float]]]:
        """
        Dự đoán ảnh từ URL

        Args:
            image_url: URL của ảnh cần dự đoán

        Returns:
            Tuple: (predicted_label_id, predicted_class_name, predicted_probability, all_probabilities)
        """
        # Tự động initialize model nếu chưa load
        if self.current_model is None:
            success = await self.initialize_model()
            if not success:
                print("⚠️ Không thể load model")
                return None, None, None, None

        try:
            # Download ảnh từ URL với timeout
            timeout = 30  # 30 seconds timeout
            response = requests.get(image_url, stream=True, timeout=timeout)
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code} khi download ảnh")
                return None, None, None, None

            # Mở ảnh bằng PIL
            img = Image.open(response.raw)
            # img.save("test.jpg")
            image = img.convert("RGB")

            # Apply transform
            image = self.transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(self.device)

            # Predict
            self.current_model.eval()
            with torch.no_grad():
                outputs = self.current_model(image)

                # Handle InceptionV3 auxiliary output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predicted_probability, predicted_label_id = torch.max(probabilities, 1)

            # Get class name
            predicted_class_name = self.class_names[predicted_label_id.item()]

            # Convert all probabilities to list
            all_probabilities = probabilities[0].cpu().numpy().tolist()

            # Cleanup tensors để tiết kiệm memory
            del image, outputs, probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return (
                predicted_label_id.item(),
                predicted_class_name,
                predicted_probability.item(),
                all_probabilities,
            )

        except requests.exceptions.Timeout:
            print("⏰ Timeout khi download ảnh")
            return None, None, None, None
        except Exception as e:
            print(f"❌ Lỗi predict: {e}")
            return None, None, None, None

    async def predict(self, image_url: str) -> List[float]:
        """
        Predict và trả về mảng xác suất cho 6 lớp

        Args:
            image_url: URL của ảnh

        Returns:
            List[float]: Mảng xác suất 6 phần tử tương ứng với lớp 0-5
        """

        if not self.is_model_loaded():
            print("❌ Model chưa được load, không thể dự đoán")
            raise Exception("Model chưa được load")

        _, _, _, all_probabilities = await self.predict_image(image_url)

        if all_probabilities is None:
            # Fallback: equal probability
            return [0.16, 0.17, 0.17, 0.17, 0.16, 0.17]

        return all_probabilities

    async def get_prediction_details(self, image_url: str) -> Dict:
        """
        Lấy thông tin dự đoán chi tiết

        Args:
            image_url: URL của ảnh

        Returns:
            Dict: Thông tin dự đoán chi tiết
        """
        predicted_id, predicted_class, predicted_prob, all_probs = (
            await self.predict_image(image_url)
        )

        if predicted_id is None:
            return {"success": False, "error": "Không thể dự đoán ảnh"}

        # Map class names to BI-RADS
        bi_rads_names = [
            "BI-RADS 0 - Cần đánh giá thêm",
            "BI-RADS 1 - Âm tính",
            "BI-RADS 2 - Tổn thương lành tính",
            "BI-RADS 3 - Có thể lành tính",
            "BI-RADS 4 - Nghi ngờ ác tính",
            "BI-RADS 5 - Rất nghi ngờ ác tính",
        ]

        # Tạo probability distribution
        probability_distribution = []
        for i, prob in enumerate(all_probs):
            probability_distribution.append(
                {
                    "class_id": i,
                    "class_name": bi_rads_names[i],
                    "probability": prob,
                    "percentage": prob * 100,
                }
            )

        # Xác định risk level
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
            "confidence": predicted_prob,
            "probability_distribution": probability_distribution,
            "risk_level": risk_level,
            "recommendation": recommendation,
        }

    def is_model_loaded(self) -> bool:
        """
        Kiểm tra model đã được load chưa
        """
        return self.current_model is not None

    async def get_model_info(self) -> Dict:
        """
        Lấy thông tin model hiện tại
        """
        # Query database để lấy thông tin model với is_active = true
        model_info = await models_collection.find_one({"is_active": True})
        if not model_info:
            print("❌ Không tìm thấy model active trong database")
            return None

        # print(f"📋 Model info từ database: {model_info}")
        return model_info

    async def initialize_model(self):
        """
        Initialize và load model khi cần thiết
        """
        if self.current_model is not None:
            print("✅ Model đã được load")
            return True

        current_model = await self.get_model_info()
        if current_model:
            print(f"🔄 Loading model: {current_model['name']}")
            return await self.load_model(current_model)
        else:
            return False


# Singleton instance
model_ai = ModelAI()


# Function để initialize model khi service start
async def initialize_model_service():
    """
    Initialize model khi service khởi động
    """
    print("🚀 Initializing model service...")

    result = await model_ai.initialize_model()
    if result:
        print("✅ Model service initialized successfully")
        return True
    else:
        print("❌ Failed to initialize model service")
        return False


# Hàm test
async def test_model_ai():
    """
    Test ModelAI với model local
    """
    print("🧪 Testing ModelAI...")

    # Initialize model service
    success = await initialize_model_service()
    if success:
        print("✅ Model đã được load")
        model_info = await model_ai.get_model_info()
        print(f"📋 Model info: {model_info}")

        # Test với sample image URL
        sample_url = "https://images.ctfassets.net/yixw23k2v6vo/duh3MNzrTx98KdhQj9iSL/e5bd997a82a5647766c62aa5e8924c1e/large-multifocal-breast-cancer-3000x2000.jpg"

        # Test predict
        probabilities = await model_ai.predict(sample_url)
        print(f"📊 Probabilities: {[f'{p:.3f}' for p in probabilities]}")

        # # Test detailed prediction
        # details = await model_ai.get_prediction_details(sample_url)
        # print(f"📋 Prediction details: {details}")

    else:
        print("❌ Model chưa được load")
        print("💡 Hãy đặt model vào backend/ai_model/best_checkpoint.pt")


if __name__ == "__main__":
    asyncio.run(test_model_ai())
