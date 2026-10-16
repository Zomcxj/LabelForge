"""视频抽帧 UI：弹窗配置、后台任务生命周期、完成后加载为数据集。

工具栏的胶片图标打开配置弹窗（按钮位置不变）；弹窗按「开始」只发信号，
任务由这里起后台线程。**弹窗在任务期间保持打开**并显示底部进度条，参数锁住；
跑完（成功、失败或取消）由这里 `accept()` 关掉，结果再报给用户。
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
        """打开视频弹窗（抽帧 / 合成两个标签页）；按用户动作起对应任务。

        弹窗不在这里关闭：任务跑完由 `_close_video_dialog` 关，这样进度条
        能一直显示在用户眼前。
        """
        if not self._cv2_available():
            self._video_warn(tr("未安装 opencv（cv2），视频功能不可用"))
            return
        from ..video_dialog import ACTION_COMPOSE, VideoDialog

        dialog = VideoDialog(self)
        # 推理成片页要列模型的类别；弹窗每次新建，所以每次灌一遍
        dialog.set_detector_info(self._all_class_names(),
                                 getattr(getattr(self, '_detector', None),
                                         'model_path', ''))
        dialog.start_requested.connect(self._on_video_start_requested)
        dialog.cancel_requested.connect(self._cancel_video_task)
        self._video_dialog = dialog
        try:
            dialog.exec_()
        finally:
            self._video_dialog = None

    def _on_video_start_requested(self, action, settings):
        """弹窗按下「开始」：起对应任务；起不来就把原因写回弹窗提示区。"""
        from ..video_dialog import ACTION_COMPOSE, ACTION_RENDER

        if getattr(self, '_busy', False):
            self._video_dialog_hint(tr("已有任务在运行，请稍候"))
            return
        dialog = self._video_dialog
        if action == ACTION_RENDER:
            started = self._start_video_render(settings)
        elif action == ACTION_COMPOSE:
            started = self._start_video_compose(settings)
        else:
            started = self._start_video_extract(settings)
        if not started:
            self._video_dialog_hint(tr("任务启动失败"))
            return
        if dialog is not None:
            dialog.begin_run()

    def _video_dialog_hint(self, message):
        dialog = getattr(self, '_video_dialog', None)
        if dialog is not None:
            dialog.show_hint(message)

    def _close_video_dialog(self, result_text=None):
        """任务收尾：弹窗里停进度、关掉窗口，再由调用方报结果。"""
        dialog = getattr(self, '_video_dialog', None)
        if dialog is None:
            return
        if result_text:
            dialog.progress_label.setText(result_text)
        dialog.set_running(False)
        dialog.accept()

    def _cancel_video_task(self):
        """弹窗上按了「取消」：取消当前在跑的任务（弹窗留着显示进度）。"""
        worker = (getattr(self, '_video_worker', None)
                  or getattr(self, '_compose_worker', None)
                  or getattr(self, '_render_worker', None))
        if worker is not None:
            worker.cancel()

    # ---------- 后台任务 ----------

    def _start_video_extract(self, settings):
        """起后台抽帧线程；返回是否真的起了。"""
        if getattr(self, '_busy', False) or getattr(self, '_video_worker', None):
            return False
        from ..inference_workers import VideoExtractWorker

        worker = VideoExtractWorker(
            settings["video_path"], settings["target_fps"],
            start=settings.get("start"), end=settings.get("end"),
            image_format=settings.get("image_format", "jpg"),
            quality=settings.get("quality", 95), parent=self)
        self._video_worker = worker
        self._video_settings = dict(settings)
        self._busy = True
        worker.progress.connect(self._on_video_progress)
        worker.finished_ok.connect(self._on_video_extract_finished)
        worker.failed.connect(self._on_video_extract_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_video_worker(w))
        worker.start()
        return True

    def _on_video_progress(self, done, total):
        """进度只进弹窗；主窗口状态栏不再显示「抽帧中 x/x」。"""
        dialog = getattr(self, '_video_dialog', None)
        if dialog is not None:
            dialog.set_progress(done, total)

    def _on_video_extract_finished(self, result):
        self._busy = False
        manifest = result.manifest
        count = len(manifest.frames)
        if result.cancelled:
            text = tr("已取消，保留已抽出的图片")
        elif not manifest.complete:
            text = tr("清单未完成，已抽出的图片保留")
        else:
            text = f"{tr('抽帧完成')} {count} {tr('张图片')}"
        self._close_video_dialog(text)
        self.status_label.setText(text)

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
            # 字号不在这里定：留 0 让引擎按设置里的「类别名字号」算
            # （`video_render.label_font_pixels()`），与画布观感一致。
        )

    def _start_video_compose(self, settings):
        """起后台合成线程；返回是否真的起了。完成后只报状态，不加载数据集。"""
        if getattr(self, '_busy', False) or getattr(self, '_compose_worker', None):
            return False
        from ..inference_workers import VideoComposeWorker

        render = self._compose_render_spec(settings.get("render"))
        worker = VideoComposeWorker(
            settings["output_path"],
            manifest_path=settings.get("manifest_path"),
            images_dir=settings.get("images_dir"),
            output_fps=settings.get("output_fps"),
            render=render, parent=self,
            start_index=settings.get("start_index"),
            end_index=settings.get("end_index"))
        self._compose_worker = worker
        self._compose_settings = dict(settings)
        self._busy = True
        worker.progress.connect(self._on_compose_progress)
        worker.finished_ok.connect(self._on_compose_finished)
        worker.failed.connect(self._on_compose_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_compose_worker(w))
        worker.start()
        return True

    def _on_compose_progress(self, done, total):
        """进度只进弹窗；主窗口状态栏不再显示「合成中 x/x」。"""
        dialog = getattr(self, '_video_dialog', None)
        if dialog is not None:
            dialog.set_progress(done, total)

    def _on_compose_finished(self, result):
        self._busy = False
        if result.cancelled:
            self._close_video_dialog(tr("已取消合成"))
            self.status_label.setText(tr("已取消合成"))
            return
        text = (f"{tr('合成完成')} {result.image_count} {tr('张图片')}  "
                f"{result.duration:.3f} {tr('秒')}  {result.output_fps:.3f} FPS")
        self._close_video_dialog(tr("合成完成"))
        self.status_label.setText(text)
        # 完成提示合并成一条：无标注张数塞进同一句话，不再单独弹窗打断
        # （有帧没有标注时画面是干净的，不说明的话用户会以为漏画了）
        message = f"{tr('输出文件')}: {result.output_path}"
        if result.unlabeled_count:
            message += (f"\n{result.unlabeled_count} "
                        f"{tr('张图片没有标注，已按原画面写入')}")
        self._video_info(message)

    def _on_compose_failed(self, message):
        self._busy = False
        self._close_video_dialog(tr("合成失败"))
        self.status_label.setText(tr("合成失败"))
        self._video_warn(message)

    def _cleanup_compose_worker(self, worker):
        if worker is getattr(self, '_compose_worker', None):
            self._compose_worker = None
        self._busy = False
        worker.deleteLater()

    def _cancel_video_compose(self):
        """请求取消；线程会在下一个采样点退出。"""
        worker = getattr(self, '_compose_worker', None)
        if worker is not None:
            worker.cancel()
            self.status_label.setText(tr("正在取消..."))

    def _video_info(self, message):
        from ..dialog_helpers import information
        information(self, tr("完成"), message)

    # ---------- 推理成片（④A） ----------

    def _start_video_render(self, settings):
        """起后台逐帧推理编码线程；返回是否真的起了。

        需要已加载的模型：没模型就明确提示，不静默什么都不做。类别色与
        范围都在这里（主线程）定稿，后台只负责逐帧跑。
        """
        if getattr(self, '_busy', False) or getattr(self, '_render_worker', None):
            return False
        detector = getattr(self, '_detector', None)
        if detector is None:
            self._video_dialog_hint(tr("请先加载模型"))
            return False
        from ..inference_workers import VideoRenderWorker

        # 范围非法（没画 ROI、窗口参数坏）在启动时拦下，不进后台跑出 0 框结果
        try:
            scope = self._scope_for_run()
        except Exception as exc:
            self._video_dialog_hint(str(exc))
            return False

        render = self._compose_render_spec(settings.get("render"))
        worker = VideoRenderWorker(
            settings["video_path"], detector,
            output_path=settings["output_path"],
            scope=scope, render=render,
            class_filter=self._filter_for_run(), parent=self)
        self._render_worker = worker
        self._render_settings = dict(settings)
        self._busy = True
        worker.progress.connect(self._on_render_progress)
        worker.finished_ok.connect(self._on_render_finished)
        worker.failed.connect(self._on_render_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_render_worker(w))
        worker.start()
        return True

    def _on_render_progress(self, done, total):
        """进度只进弹窗；主窗口状态栏不再显示「推理成片中 x/x」。"""
        dialog = getattr(self, '_video_dialog', None)
        if dialog is not None:
            dialog.set_progress(done, total)

    def _on_render_finished(self, result):
        self._busy = False
        if result.cancelled:
            # ④A 最慢（18 秒视频要推理 541 次），跑完的帧不白费：引擎会留一个
            # 能播的半成品。报出它在哪、有多长，用户能直接用，不用从头再来。
            partial = getattr(result, "partial_path", "")
            self._close_video_dialog(tr("已取消推理成片"))
            if partial:
                self.status_label.setText(
                    f"{tr('已取消推理成片')} "
                    f"{result.frames_written} {tr('帧')}")
                self._video_info(
                    f"{tr('已取消，保留已完成的部分')}: {partial}\n"
                    f"{result.frames_written} {tr('帧')}  "
                    f"{tr('正式输出未生成')}")
            else:
                self.status_label.setText(tr("已取消推理成片"))
            return
        text = (f"{tr('推理成片完成')} {result.frame_count} {tr('帧')}  "
                f"{result.duration:.3f} {tr('秒')}")
        self._close_video_dialog(tr("推理成片完成"))
        self.status_label.setText(text)
        # 框已经画进像素了，明确说清不能再改，免得用户回头找标注
        self._video_info(
            f"{tr('输出文件')}: {result.output_path}\n"
            f"{tr('框已烧进画面，不能再改')}")

    def _on_render_failed(self, message):
        self._busy = False
        self._close_video_dialog(tr("推理成片失败"))
        self.status_label.setText(tr("推理成片失败"))
        self._video_warn(message)

    def _cleanup_render_worker(self, worker):
        if worker is getattr(self, '_render_worker', None):
            self._render_worker = None
        self._busy = False
        worker.deleteLater()

    def _cancel_video_render(self):
        """请求取消；线程会在下一帧退出。"""
        worker = getattr(self, '_render_worker', None)
        if worker is not None:
            worker.cancel()
            self.status_label.setText(tr("正在取消..."))

