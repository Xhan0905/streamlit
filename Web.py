import sys
# sys.path.append(r"E:\anaconda3\envs\YOLOT\Lib\site-packages")
import os
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import tempfile
import hashlib
import random
import string
import oss2
from oss2.exceptions import OssError


def handler(event, context):
    os.system("streamlit run Web.py --server.port 8080")

# 阿里云OSS的密钥信息
ACCESS_KEY_ID = 'LTAI5tPdvSTFn4gpa4bpz4Hj'
ACCESS_KEY_SECRET = 'jef9v75IXKHxNLq3DfsTpi2Ee9Hq6U'
BUCKET_NAME = 'tjdx-tds-beta1'
ENDPOINT = 'http://oss-cn-shanghai.aliyuncs.com'

# 初始化 OSS 客户端
def init_oss_client():
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

# 上传文件到 OSS
def upload_to_oss(oss_client, file_path, object_name):
    try:
        oss_client.put_object_from_file(object_name, file_path)
        st.success(f"文件已成功上传到OSS: {object_name}")
    except OssError as e:
        st.error(f"上传到OSS时出错: {e}")
    except Exception as e:
        st.error(f"上传到OSS时出错: {e}")

# 定义保存路径
UPLOAD_FOLDER = "tjdx/uploads"
DETECTION_FOLDER = "tjdx/detections"

# 初始化保存路径
def init_folders():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(DETECTION_FOLDER):
        os.makedirs(DETECTION_FOLDER)


# 常量定义
WINDOW_TITLE = "目标检测系统（TDS_V.0.1）"
WELCOME_SENTENCE = "欢迎使用基于YOLO的目标检测与分割系统！\n同济大学徐晨团队"
USERS_FILE = "users.txt"  # 用户信息文件

# 初始化session state
def init_session():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "未加载模型"
    if 'is_paused' not in st.session_state:
        st.session_state.is_paused = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'video_writer' not in st.session_state:
        st.session_state.video_writer = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'redirect' not in st.session_state:
        st.session_state.redirect = False
    if 'captcha' not in st.session_state:
        st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# 模型加载装饰器
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 视频处理器类（用于WebRTC摄像头）
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = st.session_state.model
        self.is_paused = False

    def pause_toggle(self):
        self.is_paused = not self.is_paused

    def recv(self, frame):
        if self.is_paused:
            return frame

        img = frame.to_ndarray(format="bgr24")

        # 执行检测
        results = self.model(img)

        # 绘制结果
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

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 读取用户信息
def load_users():
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 6:
                    username, hashed_password, phone, name, unit, email = parts
                    users[username] = {
                        "password": hashed_password,
                        "phone": phone,
                        "name": name,
                        "unit": unit,
                        "email": email
                    }
                else:
                    # 如果格式不符合预期，跳过该行并记录错误
                    st.error(f"用户信息格式错误: {line.strip()}")
    return users

# 保存用户信息
def save_users(users):
    with open(USERS_FILE, "w") as f:
        for username, user_info in users.items():
            f.write(f"{username},{user_info['password']},{user_info['phone']},{user_info['name']},{user_info['unit']},{user_info['email']}\n")

# 登录页面
def login_page():
    users = load_users()
    st.title("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    captcha = st.text_input("验证码", key="captcha_input")
    st.text(f"验证码: {st.session_state.captcha}")

    if st.button("登录"):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if username in users and users[username]["password"] == hashed_password and captcha == st.session_state.captcha:
            st.session_state.logged_in = True
            st.success("登录成功！")
            st.rerun()
        else:
            st.error("用户名、密码或验证码错误！")
            st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# 注册页面
def register_page():
    users = load_users()
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
            save_users(users)
            st.success("注册成功！请登录。")
            st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# 主页
def home_page():
    st.title(WINDOW_TITLE)
    st.subheader(WELCOME_SENTENCE)

    # 封面图片
    if os.path.exists("cover.jpg"):
        st.image("cover.jpg", use_column_width=True)

    st.markdown(f"**当前模型:** {st.session_state.current_model}")

    # 模型切换
    with st.expander("模型管理"):
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("上传YOLO模型(.pt)", type=["pt"])
            if model_file and st.button("加载模型"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                        tmp.write(model_file.getbuffer())
                        st.session_state.model = load_model(tmp.name)
                        st.session_state.current_model = model_file.name
                    st.success(f"模型加载成功: {model_file.name}")
                except Exception as e:
                    st.error(f"模型加载失败: {str(e)}")

        with col2:
            if st.session_state.model and st.button("卸载模型"):
                st.session_state.model = None
                st.session_state.current_model = "未加载模型"
                st.success("模型已卸载")

    # 历史记录
    if st.button("查看历史记录"):
        if os.path.exists("records.txt"):
            with open("records.txt", "r") as f:
                st.text_area("历史记录", f.read(), height=200)
        else:
            st.warning("没有找到历史记录文件")

# 图片检测页
def image_detection(oss_client):
    st.title("图片检测")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            # 保存上传的图片
            upload_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"图片已保存到: {upload_path}")

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

                    # 保存检测结果
                    detection_path = os.path.join(DETECTION_FOLDER, f"detected_{uploaded_file.name}")
                    cv2.imwrite(detection_path, orig_img)
                    st.success(f"检测结果已保存到: {detection_path}")

                    # 上传检测结果到 OSS
                    upload_to_oss(oss_client, detection_path, f"detections/detected_{uploaded_file.name}")

                    # 显示检测结果
                    detected_image = Image.open(detection_path)
                    st.image(detected_image, caption="检测结果", use_column_width=True)

                    # 记录结果
                    detection_time = time.time() - start_time
                    st.success(f"检测完成! 耗时: {detection_time:.2f}秒 | 检测到 {len(results)} 个目标")

                    # 保存记录
                    with open("records.txt", "a") as f:
                        f.write(f"{time.ctime()} | 图片检测 | 用户: {st.session_state.username} | 用时: {detection_time:.2f}s | 目标数: {len(results)}\n")

                except Exception as e:
                    st.error(f"检测出错: {e}")

# 视频检测页
def video_detection(oss_client):
    st.title("视频检测")

    tab1, tab2, tab3 = st.tabs(["摄像头检测", "视频文件检测", "IP摄像头检测"])

    with tab2:
        st.header("视频文件检测")
        video_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])

        if video_file and st.session_state.model:
            # 保存上传的视频
            upload_path = os.path.join(UPLOAD_FOLDER, video_file.name)
            with open(upload_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.success(f"视频已保存到: {upload_path}")

            # 上传到 OSS
            upload_to_oss(oss_client, upload_path, f"uploads/{video_file.name}")

            # 视频检测
            if st.button("开始视频检测"):
                cap = cv2.VideoCapture(upload_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 保存检测结果
                detection_path = os.path.join(DETECTION_FOLDER, f"detected_{video_file.name}")
                out = cv2.VideoWriter(detection_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                frame_placeholder = st.empty()
                stop_button = st.button("停止检测")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("视频结束")
                        break

                    # 执行检测
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

                st.success(f"检测结果已保存到: {detection_path}")

# 主应用
def main():
    init_session()
    init_folders()  # 初始化保存路径
    oss_client = init_oss_client()  # 初始化 OSS 客户端

    if not st.session_state.logged_in:
        st.sidebar.title("导航")
        page = st.sidebar.radio("选择功能", ["登录", "注册"])
        if page == "登录":
            login_page()
        elif page == "注册":
            register_page()
    else:
        st.sidebar.title("导航")
        page = st.sidebar.radio("选择功能", ["主页", "图片检测", "视频检测", "退出登录"])
        if page == "主页":
            home_page()
        elif page == "图片检测":
            image_detection(oss_client)
        elif page == "视频检测":
            video_detection(oss_client)
        elif page == "退出登录":
            st.session_state.logged_in = False
            st.success("您已成功退出登录！")
            login_page()

if __name__ == "__main__":
    main()
