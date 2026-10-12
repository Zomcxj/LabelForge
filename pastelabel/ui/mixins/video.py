"""视频抽帧 UI：弹窗配置、后台任务生命周期、完成后加载为数据集。

工具栏的胶片图标打开配置弹窗（按钮位置不变）；弹窗校验通过后由这里起后台
抽帧线程，完成后把输出目录里的图片加载为数据集。
"""
from ..i18n import t as tr


class VideoMixin:
    def _cv2_available(self):
        """cv2 是否可用；缺失时视频入口置灰并说明原因。"""
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    # ---------- 配置弹窗 ----------

    def _open_video_extract(self):
        """打开视频弹窗（抽帧 / 合成两个标签页）；按用户动作起对应任务。"""
        if not self._cv2_available():
            self._video_warn(tr("未安装 opencv（cv2），视频功能不可用"))
            return
        from ..video_dialog import ACTION_COMPOSE, VideoDialog

        dialog = VideoDialog(self)
        if dialog.exec_() != dialog.Accepted:
            return
        if dialog.action == ACTION_COMPOSE:
            self._start_video_compose(dialog.settings())
        else:
            self._start_video_extract(dialog.settings())

    # ---------- 后台任务 ----------

    def _start_video_extract(self, settings):
        """起后台抽帧线程；完成后按结果加载数据集或报错。"""
        if getattr(self, '_busy', False) or getattr(self, '_video_worker', None):
            return
        from ..inference_workers import VideoExtractWorker

        worker = VideoExtractWorker(
            settings["video_path"], settings["target_fps"],
            start=settings.get("start"), end=settings.get("end"),
            image_format=settings.get("image_format", "jpg"),
            quality=settings.get("quality", 95), parent=self)
        self._video_worker = worker
        self._video_settings = dict(settings)
        self._busy = True
        self.status_label.setText(tr("抽帧中"))
        worker.progress.connect(self._on_video_progress)
        worker.finished_ok.connect(self._on_video_extract_finished)
        worker.failed.connect(self._on_video_extract_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_video_worker(w))
        worker.start()

    def _on_video_progress(self, done, total):
        self.status_label.setText(f"{tr('抽帧中')} {done}/{total}")

    def _on_video_extract_finished(self, result):
        self._busy = False
        manifest = result.manifest
        count = len(manifest.frames)
        if result.cancelled:
            self.status_label.setText(tr("已取消，保留已抽出的图片"))
        elif not manifest.complete:
            self.status_label.setText(tr("清单未完成，已抽出的图片保留"))
        else:
            self.status_label.setText(f"{tr('抽帧完成')} {count} {tr('张图片')}")

        if not result.image_paths:
            return
        self._load_video_frames(result)

    def _load_video_frames(self, result):
        """把抽出的图片加载为数据集。

        走项目自带的文件夹加载入口，别自己拼：背景图列表（background_list）、
        当前索引、分批加载与标签重扫都由它负责，少一步就会出现「画布有图、
        列表里没有路径」。
        """
        from .. import i18n

        # 抽帧输出的目录名可能与上次同名（_frames/_frames_2），但内容是新抽的，
        # 直接按目录重载，不复用旧的内存状态。
        self.load_background_folder(result.output_dir)
        self.status_label.setText(
            f"{i18n.t('抽帧目录')}: {result.output_dir}")

    def _on_video_extract_failed(self, message):
        self._busy = False
        self.status_label.setText(tr("抽帧失败"))
        self._video_warn(message)

    def _cleanup_video_worker(self, worker):
        if worker is getattr(self, '_video_worker', None):
            self._video_worker = None
        self._busy = False
        worker.deleteLater()

    def _cancel_video_extract(self):
        """请求取消；线程会在下一个采样点退出。"""
        worker = getattr(self, '_video_worker', None)
        if worker is not None:
            worker.cancel()
            self.status_label.setText(tr("正在取消..."))

    def _video_warn(self, message):
        from ..dialog_helpers import warning
        warning(self, tr("错误"), message)

    # ---------- 合成 ----------

    def _compose_render_spec(self, options):
        """把弹窗的叠加选项变成 `RenderSpec`；没有选项就返回 None。

        类别色在这里（主线程）取好：`get_label_color` 会改缓存、写配置，
        后台线程不能调用它。没取到颜色的类别走渲染模块的兜底色。
        """
        if not options:
            return None
        from ...engine.video_render import RenderSpec

        labels = [str(name) for name in (options.get("labels") or [])]
        colors = {}
        getter = getattr(self, "get_label_color", None)
        if callable(getter):
            for label in labels:
                try:
                    colors[label] = getter(label)
                except Exception:
                    continue
        return RenderSpec(
            labels=set(labels),
            show_labels=bool(options.get("show_labels")),
            show_scores=bool(options.get("show_scores")),
            label_colors=colors,
        )

    def _start_video_compose(self, settings):
        """起后台合成线程；完成后只报状态，不加载为图片数据集。"""
        if getattr(self, '_busy', False) or getattr(self, '_compose_worker', None):
            return
        from ..inference_workers import VideoComposeWorker

        render = self._compose_render_spec(settings.get("render"))
        worker = VideoComposeWorker(
            settings["output_path"],
            manifest_path=settings.get("manifest_path"),
            images_dir=settings.get("images_dir"),
            output_fps=settings.get("output_fps"),
            render=render, parent=self)
        self._compose_worker = worker
        self._compose_settings = dict(settings)
        self._busy = True
        self.status_label.setText(tr("合成中"))
        worker.progress.connect(self._on_compose_progress)
        worker.finished_ok.connect(self._on_compose_finished)
        worker.failed.connect(self._on_compose_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_compose_worker(w))
        worker.start()

    def _on_compose_progress(self, done, total):
        self.status_label.setText(f"{tr('合成中')} {done}/{total}")

    def _on_compose_finished(self, result):
        self._busy = False
        if result.cancelled:
            self.status_label.setText(tr("已取消合成"))
            return
        self.status_label.setText(
            f"{tr('合成完成')} {result.image_count} {tr('张图片')}  "
            f"{result.duration:.3f} {tr('秒')}  {result.output_fps:.3f} FPS")
        self._video_info(f"{tr('输出文件')}: {result.output_path}")
        if result.unlabeled_count:
            # 有帧没有标注，画面上是干净画面；不说明的话用户会以为漏画了
            self._video_warn(
                f"{result.unlabeled_count} {tr('张图片没有标注，已按原画面写入')}")

    def _on_compose_failed(self, message):
        self._busy = False
        self.status_label.setText(tr("合成失败"))
        self._video_warn(message)

    def _cleanup_compose_worker(self, worker):
        if worker is getattr(self, '_compose_worker', None):
            self._compose_worker = None
        self._busy = False
        worker.deleteLater()

    def _cancel_video_compose(self):
        worker = getattr(self, '_compose_worker', None)
        if worker is not None:
            worker.cancel()
            self.status_label.setText(tr("正在取消..."))

    def _video_info(self, message):
        from ..dialog_helpers import information
        information(self, tr("完成"), message)

