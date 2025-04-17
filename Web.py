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

# 内置模型配置
MODEL_REPO = {
    "yolov8n (默认)": "yolov8n.pt",
    "concraseg": "best-seg.pt",
}

# 初始化全局组件
def init_components():
    """初始化所有全局组件和配置"""
    init_session_state()
    init_folders()
    return init_oss_client()

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
        'available_models': list(MODEL_REPO.keys()),  # 确保这里使用 MODEL_REPO 的键
        'selected_model': next(iter(MODEL_REPO.keys()))  # 默认选择第一个模型
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # 自动加载默认模型
    if st.session_state.model is None and not st.session_state.logged_in:
        load_builtin_model(next(iter(MODEL_REPO.keys())))

# 下载模型文件
def download_model(model_name: str) -> Optional[str]:
    """从GitHub下载内置模型"""
    if model_name not in MODEL_REPO or MODEL_REPO[model_name] is None:
        return None
        
    model_filename = MODEL_REPO[model_name]
    model_url = f"https://github.com/Xhan0905/streamlit/blob/main/{model_filename}"
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
