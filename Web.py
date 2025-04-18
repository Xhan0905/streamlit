import sys
import os
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import av
import tempfile
import hashlib
import random
import string
import oss2
from oss2.exceptions import OssError
import json
import requests
from typing import Dict, Optional

# WebRTC配置
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 阿里云OSS配置
ACCESS_KEY_ID = 'LTAI5tPdvSTFn4gpa4bpz4Hj'
ACCESS_KEY_SECRET = 'jef9v75IXKHxNLq3DfsTpi2Ee9Hq6U'
BUCKET_NAME = 'tjdx-tds-beta1'
ENDPOINT = 'http://oss-cn-shanghai.aliyuncs.com'

# 上传文件到 OSS
def upload_to_oss(oss_client, file_path, object_name):
    try:
        oss_client.put_object_from_file(object_name, file_path)
        st.success(f"文件已成功上传到OSS: {object_name}")
    except OssError as e:
        st.error(f"上传到OSS时出错: {e}")
    except Exception as e:
        st.error(f"上传到OSS时出错: {e}")

# 常量定义
WINDOW_TITLE = "目标检测系统（TDS_beta1）"
WELCOME_SENTENCE = "欢迎使用基于YOLO的目标检测与分割系统！"
OSS_USERS_FILE = "users_info.json"  # 用户信息文件存储在OSS中

# 内置模型配置
MODEL_REPO = {
    "yolo11n (默认)": "yolo11n.pt",
    "concraseg": "best-seg.pt",
}

# 初始化文件夹
def init_folders():
    """初始化所需的文件夹"""
    os.makedirs("models", exist_ok=True)  # 用于存储下载的模型
    os.makedirs("uploads", exist_ok=True)  # 用于存储用户上传的文件
    os.makedirs("detections", exist_ok=True)  # 用于存储检测结果

# 初始化session state
def init_session_state():
    """初始化所有session状态"""
    defaults = {
        'model': None,
        'current_model': "未加载模型",
        'is_paused': False,
        'cap': None,
        'video_writer': None,
        'logged_in': False,
        'redirect': False,
        'captcha': ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)),
        'username': None,
        'webrtc_ctx': None,
        'available_models': list(MODEL_REPO.keys()),
        'selected_model': next(iter(MODEL_REPO.keys()))  # 默认选择第一个模型
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # 自动加载默认模型
    if st.session_state.model is None and not st.session_state.logged_in:
        load_builtin_model(next(iter(MODEL_REPO.keys())))

# 初始化 OSS 客户端
def init_oss_client():
    """初始化 OSS 客户端"""
    try:
        auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        oss_client = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
        return oss_client
    except OssError as e:
        st.error(f"初始化OSS客户端时出错: {e}")
        return None
    except Exception as e:
        st.error(f"初始化OSS客户端时出错: {e}")
        return None

# 初始化全局组件
def init_components():
    """初始化所有全局组件和配置"""
    init_session_state()
    init_folders()
    return init_oss_client()

# 下载模型文件
def download_model(model_name: str) -> Optional[str]:
    """从GitHub下载内置模型"""
    if model_name not in MODEL_REPO or MODEL_REPO[model_name] is None:
        return None
        
    model_filename = MODEL_REPO[model_name]
    model_url = f"https://github.com/Xhan0905/streamlit/raw/refs/heads/main/{model_filename}"
    local_path = os.path.join("models", model_filename)
    
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(local_path):
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"{model_name} 下载完成")
        except Exception as e:
            st.error(f"下载 {model_name} 失败: {str(e)}")
            return None
    return local_path

# 加载模型
@st.cache_resource
def load_model(model_path: str):
    """加载 YOLO 模型"""
    return YOLO(model_path)
    
# 加载内置模型
def load_builtin_model(model_name: str):
    """加载预定义的内置模型"""
    if model_name == "自定义模型":
        return  # 特殊处理
        
    model_path = download_model(model_name)
    if model_path:
        try:
            st.session_state.model = load_model(model_path)
            st.session_state.current_model = f"内置模型: {model_name}"
            st.session_state.selected_model = model_name
            st.success(f"{model_name} 模型加载成功")
        except Exception as e:
            st.error(f"加载 {model_name} 失败: {str(e)}")

# 加载自定义模型
def load_custom_model(model_file):
    """处理用户上传的自定义模型"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(model_file.getbuffer())
            st.session_state.model = load_model(tmp.name)
            st.session_state.current_model = f"自定义模型: {model_file.name}"
            st.session_state.selected_model = "自定义模型"
            st.success(f"模型加载成功: {model_file.name}")
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")

# 模型管理界面
def model_manager():
    """模型选择和加载界面"""
    st.subheader("模型管理")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # 模型选择下拉菜单
        selected = st.selectbox(
            "选择内置模型",
            options=st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.selected_model),
            key="model_selector"
        )
        
        if st.button("加载选定模型"):
            if selected == "自定义模型":
                st.info("请在下方的自定义模型上传区域上传模型文件")
            else:
                load_builtin_model(selected)
    
    with col2:
        # 模型卸载按钮
        if st.session_state.model and st.button("卸载当前模型"):
            st.session_state.model = None
            st.session_state.current_model = "未加载模型"
            st.success("模型已卸载")
    
    # 自定义模型上传
    st.markdown("---")
    st.subheader("自定义模型上传")
    custom_model = st.file_uploader("上传YOLO模型文件(.pt)", type=["pt"])
    if custom_model and st.button("加载自定义模型"):
        load_custom_model(custom_model)
    
    # 显示当前模型信息
    st.markdown("---")
    st.markdown(f"**当前加载模型:** {st.session_state.current_model}")
    if st.session_state.model:
        st.info(f"模型类别: {st.session_state.model.names}")

# 视频处理器类
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = st.session_state.model
        self.is_paused = st.session_state.is_paused

    def recv(self, frame):
        if self.is_paused:
            return frame

        img = frame.to_ndarray(format="bgr24")
        if self.model:
            results = self.model(img)
            img = plot_results(img, results)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 结果可视化函数
def plot_results(img, results):
    """可视化检测结果"""
    for result in results:
        # 绘制边界框
        if hasattr(result, 'boxes'):
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{result.names[int(cls)]}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制分割掩码
        if hasattr(result, 'masks') and result.masks is not None:
            for mask in result.masks.xy:
                mask = mask.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [mask], (0, 255, 0))
    return img
# 从OSS加载用户信息
def load_users(oss_client):
    """从OSS加载用户数据"""
    users = {}
    try:
        # 检查文件是否存在
        if oss_client.object_exists(OSS_USERS_FILE):
            # 下载文件内容
            data = oss_client.get_object(OSS_USERS_FILE).read()
            users = json.loads(data.decode('utf-8'))
    except (OssError, json.JSONDecodeError) as e:
        st.error(f"加载用户数据失败: {e}")
    return users
    
# 保存用户信息到OSS
def save_users(oss_client, users):
    """保存用户数据到OSS"""
    try:
        # 将用户数据转为JSON字符串
        data = json.dumps(users, ensure_ascii=False).encode('utf-8')
        # 上传到OSS（覆盖写入）
        oss_client.put_object(OSS_USERS_FILE, data)
    except OssError as e:
        st.error(f"保存用户数据到OSS失败: {e}")

# 登录页面
def login_page(oss_client):
    users = load_users(oss_client)
    st.title("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    captcha = st.text_input("验证码", key="captcha_input")
    st.text(f"验证码: {st.session_state.captcha}")

    if st.button("登录"):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if username in users and users[username]["password"] == hashed_password and captcha == st.session_state.captcha:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("登录成功！")
            st.rerun()
        else:
            st.error("用户名、密码或验证码错误！")
            st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# 注册页面
def register_page(oss_client):
    users = load_users(oss_client)
    st.title("注册")
    new_username = st.text_input("新用户名")
    new_password = st.text_input("新密码", type="password")
    new_phone = st.text_input("手机号码")
    new_name = st.text_input("姓名")
    new_unit = st.text_input("单位")
    new_email = st.text_input("邮箱")
    captcha = st.text_input("验证码", key="captcha_input")
    st.text(f"验证码: {st.session_state.captcha}")

    if st.button("注册"):
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        if new_username in users:
            st.error("用户名已存在！")
        elif captcha != st.session_state.captcha:
            st.error("验证码错误！")
        else:
            users[new_username] = {
                "password": hashed_password,
                "phone": new_phone,
                "name": new_name,
                "unit": new_unit,
                "email": new_email
            }
            save_users(oss_client, users)
            st.success("注册成功！请登录。")
            st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# 主页
def home_page():
    st.title(WINDOW_TITLE)
    st.subheader(WELCOME_SENTENCE)

    cover_image_url = "https://github.com/Xhan0905/streamlit/raw/refs/heads/main/cover.jpg"
    st.image(cover_image_url, use_column_width=True)

    st.markdown(f"**当前模型:** {st.session_state.current_model}")
    st.sidebar.markdown("### 联系我们！")
    st.sidebar.image("https://github.com/Xhan0905/streamlit/raw/refs/heads/main/2c95c15e870cffa43ff2fc977876bc0.png", width=150)
    st.sidebar.markdown("邮箱: Xhan_0905@tongji.edu.cn")
# 图片检测页
def image_detection(oss_client):
    st.title("图片检测")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            # 使用临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                upload_path = tmp.name

            # 上传到 OSS
            upload_to_oss(oss_client, upload_path, f"uploads/{uploaded_file.name}")

            # 显示原始图片
            image = Image.open(uploaded_file)
            st.image(image, caption="原始图片", use_column_width=True)

    with col2:
        if uploaded_file and st.session_state.model and st.button("开始检测"):
            with st.spinner("检测中..."):
                start_time = time.time()

                try:
                    # 执行检测
                    results = st.session_state.model(upload_path)

                    # 处理检测结果
                    orig_img = cv2.imread(upload_path)
                    for result in results:
                        # 绘制边界框
                        if hasattr(result, 'boxes'):
                            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                                x1, y1, x2, y2 = map(int, box[:4])
                                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                label = f"{result.names[int(cls)]}: {conf:.2f}"
                                cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # 保存检测结果到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        detection_path = tmp.name
                        cv2.imwrite(detection_path, orig_img)
                        
                        # 上传检测结果到 OSS
                        upload_to_oss(oss_client, detection_path, f"detections/detected_{uploaded_file.name}")

                        # 显示检测结果
                        detected_image = Image.open(detection_path)
                        st.image(detected_image, caption="检测结果", use_column_width=True)

                    # 记录结果
                    detection_time = time.time() - start_time
                    st.success(f"检测完成! 耗时: {detection_time:.2f}秒 | 检测到 {len(results)} 个目标")

                except Exception as e:
                    st.error(f"检测出错: {e}")
                finally:
                    # 删除临时文件
                    if os.path.exists(upload_path):
                        os.unlink(upload_path)
                    if 'detection_path' in locals() and os.path.exists(detection_path):
                        os.unlink(detection_path)

# 视频检测页
def video_detection(oss_client):
    st.title("视频检测")

    tab1, tab2, tab3 = st.tabs(["摄像头检测", "视频文件检测", "IP摄像头检测"])

    with tab1:
        st.header("摄像头实时检测")
        if st.session_state.model:
            st.session_state.webrtc_ctx = webrtc_streamer(
                key="example",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                async_processing=True,
                media_stream_constraints={
                    "video": True,
                    "audio": False
                }
            )
            if st.session_state.webrtc_ctx and st.session_state.webrtc_ctx.video_processor:
                if st.button("暂停/继续检测"):
                    st.session_state.webrtc_ctx.video_processor.pause_toggle()
        else:
            st.warning("请先加载模型")

    with tab2:
        st.header("视频文件检测")
        video_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])

        if video_file and st.session_state.model:
            # 使用临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
                tmp.write(video_file.read())
                upload_path = tmp.name

            # 上传到 OSS
            upload_to_oss(oss_client, upload_path, f"uploads/{video_file.name}")

            # 视频检测
            if st.button("开始视频检测"):
                cap = cv2.VideoCapture(upload_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 使用临时文件保存检测结果
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    detection_path = tmp.name
                    out = cv2.VideoWriter(detection_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                    frame_placeholder = st.empty()
                    stop_button = st.button("停止检测")

                    while cap.isOpened() and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("视频结束")
                            break

                        # 执行检测
                        if st.session_state.model:
                            results = st.session_state.model(frame)
                            # 绘制结果
                            for result in results:
                                if hasattr(result, 'boxes'):
                                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                                        x1, y1, x2, y2 = map(int, box[:4])
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                        label = f"{result.names[int(cls)]}: {conf:.2f}"
                                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        # 写入检测结果
                        out.write(frame)

                        # 显示帧
                        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

                    cap.release()
                    out.release()

                    # 上传检测结果到 OSS
                    upload_to_oss(oss_client, detection_path, f"detections/detected_{video_file.name}")

                    st.success(f"检测结果已保存到OSS")

                # 删除临时文件
                if os.path.exists(upload_path):
                    os.unlink(upload_path)
                if os.path.exists(detection_path):
                    os.unlink(detection_path)

    with tab3:
        st.header("IP摄像头检测")
        st.warning("此功能需要公开可访问的RTSP流地址")
        rtsp_url = st.text_input("输入RTSP流地址（如：rtsp://username:password@ip:port/stream）")
        
        if rtsp_url and st.session_state.model and st.button("开始检测"):
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                st.error("无法连接IP摄像头，请检查RTSP地址")
            else:
                frame_placeholder = st.empty()
                stop_button = st.button("停止检测")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("视频流中断")
                        break

                    # 执行检测
                    if st.session_state.model:
                        results = st.session_state.model(frame)
                        # 绘制结果
                        for result in results:
                            if hasattr(result, 'boxes'):
                                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                                    x1, y1, x2, y2 = map(int, box[:4])
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    label = f"{result.names[int(cls)]}: {conf:.2f}"
                                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # 显示帧
                    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

                cap.release()

# 主应用逻辑
def main():
    oss_client = init_components()
    
    if not st.session_state.logged_in:
        show_auth_pages(oss_client)
    else:
        show_main_pages(oss_client)

# 认证页面
def show_auth_pages(oss_client):
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择功能", ["登录", "注册"])
    
    if page == "登录":
        login_page(oss_client)
    elif page == "注册":
        register_page(oss_client)

# 主功能页面
def show_main_pages(oss_client):
    st.sidebar.title("导航")
    st.sidebar.markdown(f"**用户:** {st.session_state.username}")
    
    pages = {
        "主页": home_page,
        "模型管理": model_manager,
        "图片检测": lambda: image_detection(oss_client),
        "视频检测": lambda: video_detection(oss_client),
        "退出登录": logout
    }
    
    choice = st.sidebar.radio("选择功能", list(pages.keys()))
    pages[choice]()

# 退出登录
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("您已成功退出登录！")
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()
