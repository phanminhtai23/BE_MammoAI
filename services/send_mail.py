from fastapi import APIRouter
from fastapi import FastAPI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import MAILTRAP_PORT, SMTP_SERVER, MAILTRAP_LOGIN, MAILTRAP_PASSWORD, SENDER_EMAIL
import random
import string
from database import verification_codes_collection
from datetime import datetime, timezone, timedelta
from pathlib import Path

router = APIRouter()

class SMTPConnection:
    _instance = None
    _server = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SMTPConnection()
        return cls._instance
    
    def __init__(self):
        self.port = MAILTRAP_PORT
        self.smtp_server = SMTP_SERVER
        self.login = MAILTRAP_LOGIN
        self.password = MAILTRAP_PASSWORD
        self.connect()
    
    def connect(self):
        try:
            self._server = smtplib.SMTP(self.smtp_server, self.port)
            self._server.starttls()
            self._server.login(self.login, self.password)
            print("✅ Kết nối SMTP thành công")
        except Exception as e:
            print(f"❌ Lỗi kết nối SMTP: {str(e)}")
            self._server = None
    
    def send_mail(self, sender_email, receiver_emails, message):
        try:
            if self._server is None:
                self.connect()
            
            self._server.sendmail(sender_email, receiver_emails, message.as_string())
            return True
        except Exception as e:
            print(f"❌ Lỗi gửi mail: {str(e)}")
            # Thử kết nối lại nếu có lỗi
            try:
                self.connect()
                self._server.sendmail(sender_email, receiver_emails, message.as_string())
                return True
            except Exception as e2:
                print(f"❌ Lỗi gửi mail lần 2: {str(e2)}")
                return False

class SendMail:
    def __init__(self):
        self.port = MAILTRAP_PORT
        self.smtp_server = SMTP_SERVER
        self.mailtrap_login = MAILTRAP_LOGIN
        self.mailtrap_password = MAILTRAP_PASSWORD
        self.sender_email = SENDER_EMAIL
        self.receiver_emails = None
        self.smtp_connection = SMTPConnection.get_instance()

    # Gửi mã xác thực tài khoản đến người dùng
    def send_verification_email(self, receiver_emails, verification_code, expires_at):
        # Configuration
        receiver_emails = receiver_emails

        # Plain text content
        html = f"""
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Mã Xác Thực - MammoAI</title>
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        line-height: 1.6;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        line-height: 1.6;
                    }}
                    
                    .email-container {{
                        max-width: 600px;
                        margin: 0 auto;
                        background: #ffffff;
                        border-radius: 15px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    
                    .header {{
                        background: linear-gradient(135deg, #1976d2, #42a5f5);
                        color: white;
                        padding: 40px 30px;
                        text-align: center;
                        position: relative;
                    }}
                    
                    .header::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="white" opacity="0.1"/><circle cx="80" cy="40" r="1" fill="white" opacity="0.1"/><circle cx="40" cy="80" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                        opacity: 0.1;
                    }}
                    
                    .logo {{
                        font-size: 28px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        position: relative;
                        z-index: 1;
                    }}
                    
                    .header-subtitle {{
                        font-size: 16px;
                        opacity: 0.9;
                        position: relative;
                        z-index: 1;
                    }}
                    
                    .content {{
                        padding: 40px 30px;
                        text-align: center;
                    }}
                    
                    .greeting {{
                        font-size: 24px;
                        color: #333;
                        margin-bottom: 20px;
                        font-weight: 600;
                    }}
                    
                    .message {{
                        font-size: 16px;
                        color: #666;
                        margin-bottom: 30px;
                        line-height: 1.8;
                    }}
                    
                    .verification-section {{
                        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
                        border-radius: 15px;
                        padding: 30px;
                        margin: 30px 0;
                        border: 3px dashed #4e6cd2;
                    }}
                    
                    .verification-label {{
                        font-size: 14px;
                        color: #666;
                        margin-bottom: 10px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        font-weight: 600;
                    }}
                    
                    .verification-code {{
                        font-size: 36px;
                        font-weight: bold;
                        color: #4e6cd2;
                        letter-spacing: 8px;
                        font-family: 'Courier New', monospace;
                        margin: 15px 0;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                    }}
                    
                    .expiry-info {{
                        background: #fff3cd;
                        border: 1px solid #ffeaa7;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 20px 0;
                        font-size: 14px;
                        color: #856404;
                    }}
                    
                    .security-note {{
                        background: #f8d7da;
                        border: 1px solid #f5c6cb;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 20px 0;
                        font-size: 14px;
                        color: #721c24;
                    }}
                    
                    .cta-button {{
                        display: inline-block;
                        background: linear-gradient(135deg, #1976d2, #42a5f5);
                        color: white;
                        padding: 15px 30px;
                        text-decoration: none;
                        border-radius: 50px;
                        font-weight: 600;
                        font-size: 16px;
                        transition: all 0.3s ease;
                        box-shadow: 0 5px 15px rgba(25, 118, 210, 0.3);
                    }}
                    
                    .footer {{
                        background: #f8f9fa;
                        padding: 30px;
                        text-align: center;
                        border-top: 1px solid #e9ecef;
                    }}
                    
                    .footer-text {{
                        color: #666;
                        font-size: 14px;
                        margin-bottom: 15px;
                    }}
                    
                    .social-links {{
                        margin: 20px 0;
                    }}
                    
                    .social-links a {{
                        display: inline-block;
                        margin: 0 10px;
                        color: #4e6cd2;
                        text-decoration: none;
                        font-size: 14px;
                    }}
                    
                    .company-info {{
                        color: #999;
                        font-size: 12px;
                        margin-top: 20px;
                    }}
                    
                    .medical-icon {{
                        font-size: 48px;
                        margin-bottom: 20px;
                    }}
                    
                    @media (max-width: 600px) {{
                        .email-container {{
                            margin: 10px;
                            border-radius: 10px;
                        }}
                        
                        .header, .content, .footer {{
                            padding: 20px;
                        }}
                        
                        .verification-code {{
                            font-size: 28px;
                            letter-spacing: 4px;
                        }}
                        
                        .greeting {{
                            font-size: 20px;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="email-container">
                    <!-- Header -->
                    <div class="header">
                        <div class="medical-icon">🎗️</div>
                        <div class="logo">MammoAI</div>
                        <div class="header-subtitle">Hệ thống dự đoán ung thư vú thông minh</div>
                    </div>
                    
                    <!-- Content -->
                    <div class="content">
                        <div class="greeting">Xin chào ! 👋</div>
                        
                        <div class="message">
                            Chúng tôi đã nhận được yêu cầu xác thực tài khoản của bạn trên hệ thống 
                            <strong>MammoAI</strong>. Để hoàn tất quá trình, vui lòng sử dụng mã xác thực dưới đây:
                        </div>
                        
                        <!-- Verification Code Section -->
                        <div class="verification-section">
                            <div class="verification-label">Mã xác thực của bạn</div>
                            <div class="verification-code"> {verification_code} </div>
                            
                            <div class="expiry-info">
                                ⏰ <strong>Lưu ý:</strong> Mã này sẽ hết hạn vào lúc <strong>{expires_at.strftime("%H:%M:%S %d-%m-%Y")}</strong>
                            </div>
                        </div>
                        
                        <div class="security-note">
                            🔒 <strong>Bảo mật:</strong> Không chia sẻ mã này với bất kỳ ai. Nhân viên của chúng tôi sẽ không bao giờ yêu cầu mã xác thực qua điện thoại hoặc email.
                        </div>
                        
                        <div class="message">
                            Nếu bạn không yêu cầu mã xác thực này, vui lòng bỏ qua email này hoặc liên hệ với chúng tôi ngay lập tức.
                        </div>
                    </div>
                    
                    <!-- Footer -->
                    <div class="footer">
                        <div class="footer-text">
                            <strong>MammoAI</strong> - Công nghệ AI hỗ trợ chẩn đoán sớm
                        </div>
                        
                        <div class="social-links">
                            <a href="#">📧 Hỗ trợ</a>
                            <a href="#">🌐 Website</a>
                            <a href="#">📱 Mobile App</a>
                        </div>
                        
                        <div class="company-info">
                            © 2025 MammoAI. Tất cả quyền được bảo lưu.<br>
                            Email này được gửi tự động, vui lòng không reply.<br>
                            🏥 Phát triển bởi Phan Minh Tai
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """

        # Create MIMEText object
        message = MIMEMultipart("alternative")
        message = MIMEText(html, "html")
        message["Subject"] = f"[MammoAI] Mã xác thực tài khoản "
        message["From"] = self.sender_email
        # Join the list of receiver emails into a string separated by commas
        message["To"] = receiver_emails

        # Sử dụng kết nối SMTP đã được tạo trước đó
        if self.smtp_connection.send_mail(self.sender_email, receiver_emails, message):
            return {"msg":"send mail success"}
        else:
            return {"msg":"send mail failed"}

    # Tạo mã xác thực 6 số
    def generate_verification_code(self):
        """Tạo mã xác thực 6 số"""
        return ''.join(random.choices(string.digits, k=6))

    def send_forgot_password_email(self, receiver_emails, token, expires_at, reset_password_url):
        # Configuration
        receiver_emails = receiver_emails

        # Plain text content
        html = f"""<!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Đặt lại mật khẩu - MammoAI</title>
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        line-height: 1.6;
                    }}
                    
                    .email-container {{
                        max-width: 600px;
                        margin: 0 auto;
                        background: #ffffff;
                        border-radius: 15px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    
                    .header {{
                        background: linear-gradient(135deg, #1976d2, #42a5f5);
                        color: white;
                        padding: 40px 30px;
                        text-align: center;
                        position: relative;
                    }}
                    
                    .header::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="white" opacity="0.1"/><circle cx="80" cy="40" r="1" fill="white" opacity="0.1"/><circle cx="40" cy="80" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                        opacity: 0.1;
                    }}
                    
                    .logo {{
                        font-size: 28px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        position: relative;
                        z-index: 1;
                    }}
                    
                    .header-subtitle {{
                        font-size: 16px;
                        opacity: 0.9;
                        position: relative;
                        z-index: 1;
                    }}
                    
                    .content {{
                        padding: 40px 30px;
                        text-align: center;
                    }}
                    
                    .greeting {{
                        font-size: 24px;
                        color: #333;
                        margin-bottom: 20px;
                        font-weight: 600;
                    }}
                    
                    .message {{
                        font-size: 16px;
                        color: #666;
                        margin-bottom: 30px;
                        line-height: 1.8;
                    }}
                    
                    .verification-section {{
                        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
                        border-radius: 15px;
                        padding: 30px;
                        margin: 30px 0;
                        border: 3px dashed #4e6cd2;
                    }}
                    
                    .verification-label {{
                        font-size: 14px;
                        color: #666;
                        margin-bottom: 10px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        font-weight: 600;
                    }}
                    
                    .verification-code {{
                        font-size: 36px;
                        font-weight: bold;
                        color: #4e6cd2;
                        letter-spacing: 8px;
                        font-family: 'Courier New', monospace;
                        margin: 15px 0;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                    }}
                    
                    .expiry-info {{
                        background: #fff3cd;
                        border: 1px solid #ffeaa7;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 20px 0;
                        font-size: 14px;
                        color: #856404;
                    }}
                    
                    .security-note {{
                        background: #f8d7da;
                        border: 1px solid #f5c6cb;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 20px 0;
                        font-size: 14px;
                        color: #721c24;
                    }}
                    
                    .cta-button {{
                        display: inline-block;
                        background: linear-gradient(135deg, #1976d2, #42a5f5);
                        color: white;
                        padding: 15px 30px;
                        text-decoration: none;
                        border-radius: 50px;
                        font-weight: 600;
                        font-size: 16px;
                        transition: all 0.3s ease;
                        box-shadow: 0 5px 15px rgba(25, 118, 210, 0.3);
                    }}
                    
                    .footer {{
                        background: #f8f9fa;
                        padding: 30px;
                        text-align: center;
                        border-top: 1px solid #e9ecef;
                    }}
                    
                    .footer-text {{
                        color: #666;
                        font-size: 14px;
                        margin-bottom: 15px;
                    }}
                    
                    .social-links {{
                        margin: 20px 0;
                    }}
                    
                    .social-links a {{
                        display: inline-block;
                        margin: 0 10px;
                        color: #4e6cd2;
                        text-decoration: none;
                        font-size: 14px;
                    }}
                    
                    .company-info {{
                        color: #999;
                        font-size: 12px;
                        margin-top: 20px;
                    }}
                    
                    .medical-icon {{
                        font-size: 48px;
                        margin-bottom: 20px;
                    }}
                    
                    @media (max-width: 600px) {{
                        .email-container {{
                            margin: 10px;
                            border-radius: 10px;
                        }}
                        
                        .header, .content, .footer {{
                            padding: 20px;
                        }}
                        
                        .verification-code {{
                            font-size: 28px;
                            letter-spacing: 4px;
                        }}
                        
                        .greeting {{
                            font-size: 20px;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="email-container">
                    <!-- Header -->
                    <div class="header">
                        <div class="medical-icon">🎗️</div>
                        <div class="logo">MammoAI</div>
                        <div class="header-subtitle">Hệ thống dự đoán ung thư vú thông minh</div>
                    </div>
                    
                    <!-- Content -->
                    <div class="content">
                        <div class="greeting">Xin chào ! 👋</div>
                        
                        <div class="message">
                            Chúng tôi đã nhận được yêu cầu xác thực tài khoản của bạn trên hệ thống 
                            <strong>MammoAI</strong>. Để hoàn tất quá trình, vui lòng sử dụng mã xác thực dưới đây:
                        </div>
                        
                        <!-- Verification Code Section -->
                        <div class="verification-section">
                            <div class="verification-label">Nhấn vào link để đặt lại mật khẩu của bạn</div>
                                <a href="{reset_password_url}" class="cta-button" style="color: #f0f1f3"> Nhấp vào đây </a>
                            <div class="expiry-info">
                                ⏰ <strong>Lưu ý:</strong> Đường link này sẽ hết hạn vào lúc <strong>{expires_at.strftime("%H:%M:%S %d-%m-%Y")}</strong>
                            </div>
                        </div>
                        
                        <div class="message">
                            Nếu bạn không yêu cầu đặt lại mật khẩu, vui lòng bỏ qua email này hoặc liên hệ với chúng tôi ngay lập tức.
                        </div>
                    </div>
                    
                    <!-- Footer -->
                    <div class="footer">
                        <div class="footer-text">
                            <strong>MammoAI</strong> - Công nghệ AI hỗ trợ chẩn đoán sớm
                        </div>
                        
                        <div class="social-links">
                            <a href="#">📧 Hỗ trợ</a>
                            <a href="#">🌐 Website</a>
                            <a href="#">📱 Mobile App</a>
                        </div>
                        
                        <div class="company-info">
                            © 2025 MammoAI. Tất cả quyền được bảo lưu.<br>
                            Email này được gửi tự động, vui lòng không reply.<br>
                            🏥 Phát triển bởi Phan Minh Tai
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """

        # Create MIMEText object
        message = MIMEMultipart("alternative")
        message = MIMEText(html, "html")
        message["Subject"] = f"[MammoAI] Đặt lại mật khẩu "
        message["From"] = self.sender_email
        # Join the list of receiver emails into a string separated by commas
        message["To"] = receiver_emails

        # Sử dụng kết nối SMTP đã được tạo trước đó
        if self.smtp_connection.send_mail(self.sender_email, receiver_emails, message):
            return {"msg":"send mail success"}
        else:
            return {"msg":"send mail failed"}


# Khởi tạo đối tượng SendMail để sử dụng trong toàn bộ ứng dụng
mail_service = SendMail()

# Export các hàm cần thiết để sử dụng trong các module khác
def send_verification_email(receiver_emails, verification_code, expires_at):
    return mail_service.send_verification_email(receiver_emails, verification_code, expires_at)

def generate_verification_code():
    return mail_service.generate_verification_code()

def send_forgot_password_email(receiver_emails, token, expires_at, reset_password_url):
    return mail_service.send_forgot_password_email(receiver_emails, token, expires_at, reset_password_url)