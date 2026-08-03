"""orbbec_ctypes.py — 用 ctypes 封装本地 OrbbecSDK (v2.7.6, /opt/OrbbecSDK_v2.7.6)。

零安装依赖,直接调 C API。覆盖 orbbec_live.py 需要的子集:
  pipeline + config + video stream + D2C 对齐 + wait_for_frameset + 内参。

用法:
  cam = OrbbecCamera(width=640, height=480, fps=30)
  color, depth_u16, intrinsics = cam.grab()   # color: (H,W,3) uint8 RGB; depth: (H,W) uint16 mm
  cam.close()
"""

import ctypes
import ctypes.util
import numpy as np
from pathlib import Path

SDK_LIB = Path("/opt/OrbbecSDK_v2.7.6/lib/libOrbbecSDK.so.2.7.6")

# ---------------- 枚举 ----------------
OB_STREAM_COLOR = 2
OB_STREAM_DEPTH = 3
OB_FORMAT_Y16 = 8
OB_FORMAT_RGB = 22
ALIGN_DISABLE = 0
ALIGN_D2C_HW_MODE = 1
ALIGN_D2C_SW_MODE = 2

# ---------------- 结构体 ----------------
class ObCameraIntrinsic(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_float), ("fy", ctypes.c_float),
        ("cx", ctypes.c_float), ("cy", ctypes.c_float),
        ("width", ctypes.c_int16), ("height", ctypes.c_int16),
    ]

class ObCameraDistortion(ctypes.Structure):
    _fields_ = [
        ("k1", ctypes.c_float), ("k2", ctypes.c_float), ("k3", ctypes.c_float),
        ("k4", ctypes.c_float), ("k5", ctypes.c_float), ("k6", ctypes.c_float),
        ("p1", ctypes.c_float), ("p2", ctypes.c_float),
        ("model", ctypes.c_int),
    ]

class ObD2CTransform(ctypes.Structure):
    _fields_ = [("rot", ctypes.c_float * 9), ("trans", ctypes.c_float * 3)]

class ObCameraParam(ctypes.Structure):
    _fields_ = [
        ("depthIntrinsic", ObCameraIntrinsic),
        ("rgbIntrinsic", ObCameraIntrinsic),
        ("depthDistortion", ObCameraDistortion),
        ("rgbDistortion", ObCameraDistortion),
        ("transform", ObD2CTransform),
        ("isMirrored", ctypes.c_bool),
    ]


class _OrbbecSDK:
    """懒加载单例:ctypes 绑定 + 错误检查。"""

    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            if not SDK_LIB.exists():
                raise RuntimeError(f"OrbbecSDK 未找到: {SDK_LIB}")
            lib = ctypes.CDLL(str(SDK_LIB), mode=ctypes.RTLD_GLOBAL)
            # 函数原型
            lib.ob_create_pipeline.restype = ctypes.c_void_p
            lib.ob_create_pipeline.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_delete_pipeline.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_create_config.restype = ctypes.c_void_p
            lib.ob_create_config.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_delete_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_config_enable_video_stream.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_config_set_align_mode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_config_set_frame_aggregate_output_mode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_config_enable_stream_with_stream_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_config_enable_all_stream.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_start_with_config.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_start_with_callback.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p),
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_stop.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_wait_for_frameset.restype = ctypes.c_void_p
            lib.ob_pipeline_wait_for_frameset.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_get_camera_param.restype = ObCameraParam
            lib.ob_pipeline_get_camera_param.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_pipeline_get_stream_profile_list.restype = ctypes.c_void_p
            lib.ob_pipeline_get_stream_profile_list.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_stream_profile_list_get_count.restype = ctypes.c_uint32
            lib.ob_stream_profile_list_get_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_stream_profile_list_get_profile.restype = ctypes.c_void_p
            lib.ob_stream_profile_list_get_profile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_stream_profile_get_format.restype = ctypes.c_int
            lib.ob_stream_profile_get_format.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_stream_profile_get_type.restype = ctypes.c_int
            lib.ob_stream_profile_get_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_video_stream_profile_get_fps.restype = ctypes.c_uint32
            lib.ob_video_stream_profile_get_fps.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frameset_get_color_frame.restype = ctypes.c_void_p
            lib.ob_frameset_get_color_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frameset_get_depth_frame.restype = ctypes.c_void_p
            lib.ob_frameset_get_depth_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frameset_get_frame.restype = ctypes.c_void_p
            lib.ob_frameset_get_frame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frameset_get_count.restype = ctypes.c_uint32
            lib.ob_frameset_get_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frame_get_data.restype = ctypes.POINTER(ctypes.c_uint8)
            lib.ob_frame_get_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frame_get_data_size.restype = ctypes.c_uint32
            lib.ob_frame_get_data_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frame_get_format.restype = ctypes.c_int
            lib.ob_frame_get_format.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_frame_get_stream_profile.restype = ctypes.c_void_p
            lib.ob_frame_get_stream_profile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_video_stream_profile_get_width.restype = ctypes.c_uint32
            lib.ob_video_stream_profile_get_width.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_video_stream_profile_get_height.restype = ctypes.c_uint32
            lib.ob_video_stream_profile_get_height.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_delete_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.ob_error_get_message.restype = ctypes.c_char_p
            lib.ob_error_get_message.argtypes = [ctypes.c_void_p]
            lib.ob_delete_error.argtypes = [ctypes.c_void_p]
            cls._lib = lib
        return cls._lib

    @classmethod
    def _check(cls, errp):
        """errp: c_void_p;非空指针则抛异常。"""
        if errp.value:
            lib = cls.lib()
            msg = lib.ob_error_get_message(errp.value)
            lib.ob_delete_error(errp.value)
            raise RuntimeError(msg.decode() if msg else "OrbbecSDK error")

    @classmethod
    def call(cls, fn, *args):
        """调用 void 返回的 SDK 函数(句柄经入参输出)。"""
        errp = ctypes.c_void_p()
        fn(*args, ctypes.byref(errp))
        cls._check(errp)

    @classmethod
    def create(cls, fn):
        """调用工厂函数:句柄是返回值,ob_error** 是唯一入参。"""
        errp = ctypes.c_void_p()
        handle = fn(ctypes.byref(errp))
        cls._check(errp)
        if not handle:
            raise RuntimeError(f"{getattr(fn, '__name__', 'SDK factory')} returned NULL")
        return ctypes.c_void_p(handle)


class OrbbecCamera:
    """回调模式采集:SDK 线程把最新 color+depth 拷贝到槽位,grab() 线程安全取用。"""

    _CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)

    def __init__(self, width=640, height=480, fps=30, align=True):
        import threading
        lib = _OrbbecSDK.lib()
        self._pipe = ctypes.c_void_p()
        self._config = ctypes.c_void_p()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._latest = None
        self._started = False
        self._cb_ref = None

        def _on_frameset(frameset_ptr, _user):
            if not frameset_ptr:
                return
            try:
                errp = ctypes.c_void_p()
                cfr = lib.ob_frameset_get_color_frame(frameset_ptr, ctypes.byref(errp))
                dfr = lib.ob_frameset_get_depth_frame(frameset_ptr, ctypes.byref(errp))
                if cfr and dfr:
                    errp.value = None
                    csz = lib.ob_frame_get_data_size(cfr, ctypes.byref(errp))
                    dsz = lib.ob_frame_get_data_size(dfr, ctypes.byref(errp))
                    cfmt = lib.ob_frame_get_format(cfr, ctypes.byref(errp))
                    dfmt = lib.ob_frame_get_format(dfr, ctypes.byref(errp))
                    cd = lib.ob_frame_get_data(cfr, ctypes.byref(errp))
                    dd = lib.ob_frame_get_data(dfr, ctypes.byref(errp))
                    cw = self._intr["width"]
                    ch = self._intr["height"]
                    if dfmt == OB_FORMAT_Y16:  # Y16: 2 字节/像素,须按 uint16 读
                        dptr = ctypes.cast(dd, ctypes.POINTER(ctypes.c_uint16))
                        depth = np.ctypeslib.as_array(dptr, shape=(ch, cw)).copy()
                    else:
                        depth = np.ctypeslib.as_array(dd, shape=(ch, cw)).copy().astype(np.uint16)
                    color = np.ctypeslib.as_array(cd, shape=(ch, cw, 3)).copy()
                    with self._cv:
                        self._latest = (color, depth)
                        self._cv.notify_all()
                if cfr:
                    lib.ob_delete_frame(cfr, ctypes.byref(ctypes.c_void_p()))
                if dfr:
                    lib.ob_delete_frame(dfr, ctypes.byref(ctypes.c_void_p()))
            except Exception:
                pass
            finally:
                lib.ob_delete_frame(frameset_ptr, ctypes.byref(ctypes.c_void_p()))

        self._pipe = _OrbbecSDK.create(lib.ob_create_pipeline)
        try:
            self._config = _OrbbecSDK.create(lib.ob_create_config)
            _OrbbecSDK.call(lib.ob_config_enable_video_stream,
                            self._config, OB_STREAM_COLOR, width, height, fps, OB_FORMAT_RGB)
            _OrbbecSDK.call(lib.ob_config_enable_video_stream,
                            self._config, OB_STREAM_DEPTH, width, height, fps, OB_FORMAT_Y16)
            if align:
                _OrbbecSDK.call(lib.ob_config_set_align_mode, self._config, ALIGN_D2C_SW_MODE)
            # 回调里用请求分辨率兜底(内参在 start 后才有效)
            self._intr = {"width": int(width), "height": int(height)}
            self._cb_ref = self._CALLBACK(_on_frameset)
            _OrbbecSDK.call(lib.ob_pipeline_start_with_callback,
                            self._pipe, self._config, self._cb_ref, None)
            # start 之后内参才有效
            cam_param = lib.ob_pipeline_get_camera_param(self._pipe, ctypes.byref(ctypes.c_void_p()))
            intr = cam_param.rgbIntrinsic
            if intr.width > 0 and intr.height > 0:
                self._intr = {"width": int(intr.width), "height": int(intr.height)}
            self._started = True
        except Exception:
            self.close()
            raise

        self.intrinsics = {
            "fx": float(intr.fx), "fy": float(intr.fy),
            "cx": float(intr.cx), "cy": float(intr.cy),
            "width": self._intr["width"], "height": self._intr["height"],
        }
        print(f"[相机] ctypes SDK 回调模式已启动: {self.intrinsics['width']}x{self.intrinsics['height']} (D2C 对齐)")
        print(f"[相机] 内参 fx={self.intrinsics['fx']:.3f} fy={self.intrinsics['fy']:.3f} "
              f"cx={self.intrinsics['cx']:.3f} cy={self.intrinsics['cy']:.3f}")

    def grab(self, timeout_ms=2000):
        """返回最新一帧 (color_rgb_uint8 (H,W,3), depth_u16_mm (H,W)),超时返回 None。"""
        import time
        with self._cv:
            if self._latest is None:
                self._cv.wait(timeout_ms / 1000.0)
            got = self._latest
            self._latest = None
        return got

    def close(self):
        lib = _OrbbecSDK.lib()
        if getattr(self, "_pipe", None) and self._pipe.value:
            try:
                _OrbbecSDK.call(lib.ob_pipeline_stop, self._pipe)
            except Exception:
                pass
            try:
                _OrbbecSDK.call(lib.ob_delete_pipeline, self._pipe)
            except Exception:
                pass
            self._pipe = ctypes.c_void_p()
        if getattr(self, "_config", None) and self._config.value:
            try:
                _OrbbecSDK.call(lib.ob_delete_config, self._config)
            except Exception:
                pass
            self._config = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
