"""Tk presentation of shared clip recognition and bilingual captions."""
from __future__ import annotations
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from .captions import CaptionCatalog, caption_from_result
from .session import RecognitionSession
from .sources import CameraSource, VideoSource

class SignLanguageApp:
    def __init__(self, root, *, session=None, catalog=None, model=None, video=None, camera=None):
        self.root = root
        self.session = session if session is not None else RecognitionSession()
        self.catalog = catalog if catalog is not None else CaptionCatalog()
        self.caption = None
        self._last_result = None
        self._metadata_model = None
        self._closed = False
        self._photo = None
        self._after_id = None
        root.title('手语片段识别 | Ponte de Sinais')
        root.geometry('1060x780')
        root.minsize(720, 680)
        self.zh_var = tk.StringVar(value='等待识别')
        self.pt_var = tk.StringVar(value='')
        self.notice_var = tk.StringVar(value='')
        self.status_var = tk.StringVar(value='就绪：打开文件或摄像头')
        self.source_var = tk.StringVar(value='来源：未选择')
        self.model_var = tk.StringVar(value='模型：未加载')
        self.catalog_var = tk.StringVar(value='词典：未加载；中文释义待补充')
        self.camera_var = tk.StringVar(value='0')
        self.mirror_var = tk.BooleanVar(value=True)
        self._build_ui()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        if video:
            self.open_video(video)
        elif camera is not None:
            self.open_camera(camera)
        if model:
            self.load_model(model)
        self._poll()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='Libras 完整动作片段识别', font=('Microsoft YaHei UI', 19, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='摄像头实时预览 → 开始/结束一次动作录制 → 片段识别。当前模型不提供连续翻译。', wraplength=950).pack(anchor='w', pady=(3, 10))
        load_row = ttk.Frame(outer)
        load_row.pack(fill='x')
        self.model_button = ttk.Button(load_row, text='选择模型目录', command=self._choose_model)
        self.model_button.pack(side='left')
        self.catalog_button = ttk.Button(load_row, text='加载/重载字幕词典', command=self._choose_catalog)
        self.catalog_button.pack(side='left', padx=6)
        ttk.Label(outer, textvariable=self.model_var, wraplength=950).pack(anchor='w', pady=3)
        ttk.Label(outer, textvariable=self.catalog_var, wraplength=950).pack(anchor='w')
        self.vocabulary_label = ttk.Label(outer, text='支持词汇：尚未加载模型', wraplength=950)
        self.vocabulary_label.pack(anchor='w', pady=(0, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill='x', pady=4)
        self.video_button = ttk.Button(controls, text='打开视频', command=self._choose_video)
        self.video_button.pack(side='left')
        ttk.Label(controls, text='摄像头设备：').pack(side='left', padx=(12, 0))
        self.camera_entry = ttk.Entry(controls, width=4, textvariable=self.camera_var)
        self.camera_entry.pack(side='left')
        self.camera_button = ttk.Button(controls, text='打开摄像头', command=self.open_camera)
        self.camera_button.pack(side='left', padx=5)
        self.stop_button = ttk.Button(controls, text='关闭来源', command=self.stop_source)
        self.stop_button.pack(side='left', padx=5)
        self.mirror_check = ttk.Checkbutton(controls, text='镜像预览（不影响识别）', variable=self.mirror_var)
        self.mirror_check.pack(side='left', padx=5)
        ttk.Label(outer, textvariable=self.source_var, wraplength=950).pack(anchor='w')
        self.preview = tk.Label(outer, background='#17232d', foreground='white', text='预览', anchor='center')
        self.preview.pack(fill='both', expand=True, pady=8)
        actions = ttk.Frame(outer)
        actions.pack(fill='x')
        self.pause_button = ttk.Button(actions, text='暂停/继续文件预览', command=self.toggle_pause)
        self.pause_button.pack(side='left')
        self.recognize_button = ttk.Button(actions, text='识别完整文件', command=lambda: self._call(self.session.recognize_file))
        self.recognize_button.pack(side='left', padx=6)
        self.record_button = ttk.Button(actions, text='开始录制', command=lambda: self._call(self.session.start_recording))
        self.record_button.pack(side='left', padx=6)
        self.finish_button = ttk.Button(actions, text='结束录制并识别', command=lambda: self._call(self.session.finish_recording))
        self.finish_button.pack(side='left', padx=6)
        self.cancel_button = ttk.Button(actions, text='取消当前动作/任务', command=self.cancel)
        self.cancel_button.pack(side='left', padx=6)
        ttk.Label(outer, text='录制上限：15 秒 / 600 帧 / 128 MiB；超限请重新录制。', foreground='#555555').pack(anchor='w', pady=3)
        self.zh_label = ttk.Label(outer, textvariable=self.zh_var, font=('Microsoft YaHei UI', 24, 'bold'), wraplength=950)
        self.zh_label.pack(anchor='center', pady=(4, 0))
        self.pt_label = ttk.Label(outer, textvariable=self.pt_var, font=('Segoe UI', 18), wraplength=950)
        self.pt_label.pack(anchor='center')
        self.notice_label = ttk.Label(outer, textvariable=self.notice_var, foreground='#925300', wraplength=950)
        self.notice_label.pack(anchor='center')
        ttk.Label(outer, textvariable=self.status_var, wraplength=950).pack(anchor='w', pady=(6, 0))
        # Reserve every control/caption row; only preview absorbs/shrinks space.
        # Image labels otherwise grow their requested height and pack hides the
        # trailing subtitle widgets once a full-sized PhotoImage is displayed.
        for widget in outer.winfo_children():
            widget.pack_forget()
        for row, widget in enumerate(outer.winfo_children()):
            widget.grid(row=row, column=0, sticky='nsew' if widget is self.preview else 'ew', pady=2)
            outer.rowconfigure(row, weight=1 if widget is self.preview else 0)
        outer.columnconfigure(0, weight=1)
        for column in range(3):
            actions.columnconfigure(column, weight=1)
        buttons = (self.pause_button, self.recognize_button, self.record_button,
                   self.finish_button, self.cancel_button)
        for button in buttons:
            button.pack_forget()
        for index, button in enumerate(buttons):
            button.grid(row=index // 3, column=index % 3, sticky='ew', padx=3, pady=2)
        self.mirror_check.pack_forget()
        self.mirror_check.pack(side='bottom', anchor='w')
        self.root.bind('<Configure>', lambda event: self._resize_text(outer) if event.widget is self.root else None)

    def _resize_text(self, outer):
        width = max(300, self.root.winfo_width() - 32)
        for widget in outer.winfo_children():
            if isinstance(widget, ttk.Label):
                widget.configure(wraplength=width)

    def _clear_caption(self):
        self.caption = None
        self._last_result = None
        self.zh_var.set('等待识别')
        self.pt_var.set('')
        self.notice_var.set('')

    def _error(self, error):
        self._clear_caption()
        self.status_var.set(f'错误：{error}；可重新打开来源或重试')

    def _call(self, operation):
        try:
            value = operation()
            self._clear_caption()
            self.status_var.set(self._state_text())
            self._update_controls()
            return value
        except Exception as error:
            self._error(error)
            return None

    def _choose_video(self):
        path = filedialog.askopenfilename(title='打开完整动作视频', filetypes=[('视频', '*.mp4 *.avi *.mov *.mkv'), ('所有文件', '*.*')])
        if path:
            self.open_video(path)

    def open_video(self, path):
        self._call(lambda: self.session.open_source(VideoSource(path), kind='video', video_path=path))
        self.source_var.set(f'来源：文件 {path}')
        self._reset_preview()

    def open_camera(self, index=None):
        try:
            index = int(self.camera_var.get()) if index is None else int(index)
            if index < 0:
                raise ValueError('摄像头设备编号须为非负整数')
            self._call(lambda: self.session.open_source(CameraSource(index), kind='camera'))
            self.source_var.set(f'来源：摄像头 {index}')
            self._reset_preview()
        except Exception as error:
            self._error(error)

    def stop_source(self):
        self._call(self.session.stop_source)
        self.source_var.set('来源：已关闭')
        self.preview.configure(image='', text='预览')
        self._photo = None

    def _reset_preview(self):
        self._photo = None
        self.preview.configure(image='', text='正在打开来源…')

    def _choose_model(self):
        path = filedialog.askdirectory(title='选择含 manifest.json 的模型包目录')
        if path:
            self.load_model(path)

    def load_model(self, path):
        self._call(lambda: self.session.load_model(path))

    def _choose_catalog(self):
        path = filedialog.askopenfilename(title='加载字幕词典 JSON', filetypes=[('字幕词典', '*.json')])
        if path:
            self.load_catalog(path)

    def load_catalog(self, path):
        try:
            self.catalog = CaptionCatalog.load(path)
            self.catalog_var.set(f"词典：{Path(path).name} · {self.catalog.catalog_id}")
            if self.caption is not None and self._last_result is not None:
                self._display_result(self._last_result, self.session.request_id, self.session.epoch)
        except Exception as error:
            self._error(error)

    def cancel(self):
        self._call(self.session.cancel)
        if self.session.busy:
            self.status_var.set('已取消显示；正在等待底层模型任务结束')

    def toggle_pause(self):
        if self.session.source_kind == 'video':
            self.session.paused = not self.session.paused
            self.status_var.set('预览已暂停；完整文件识别不受影响' if self.session.paused else '继续文件预览')

    def _state_text(self):
        return {'IDLE': '就绪', 'PREVIEW': '预览中', 'RECORDING': '录制中：完成一个动作后结束录制', 'PROCESSING': '正在处理完整片段…', 'RESULT': '识别完成', 'REJECTED': '本片段未能识别', 'ERROR': '来源或任务错误'}.get(self.session.state, self.session.state)

    def _update_controls(self):
        busy = self.session.busy
        state = self.session.state
        camera = self.session.source_kind == 'camera' and self.session.source_active
        video = self.session.video_path is not None
        loaded = self.session.recognizer is not None
        for button, enabled in ((self.model_button, not busy), (self.recognize_button, video and loaded and not busy), (self.record_button, camera and loaded and not busy and state != 'RECORDING'), (self.finish_button, state == 'RECORDING' and not busy), (self.pause_button, video)):
            button.configure(state='normal' if enabled else 'disabled')

    def _display_result(self, result, request_id, epoch):
        self._last_result = result
        self.caption = caption_from_result(result, self.catalog, request_id=request_id, epoch=epoch)
        self.zh_var.set(self.caption.zh)
        self.pt_var.set(self.caption.pt)
        self.notice_var.set(self.caption.notice)
        score = f' · 分数 {self.caption.score:.3f}' if self.caption.score is not None else ''
        self.status_var.set(f"{self._state_text()} · 标签 {self.caption.label_id or '无'}{score} · 来源 {result.sample_id}")

    def _render_preview(self, packet):
        # Only this display copy is mirrored; captured input remains original BGR.
        from PIL import Image, ImageTk
        rgb = packet.image[:, :, ::-1]
        if self.session.source_kind == 'camera' and self.mirror_var.get():
            rgb = rgb[:, ::-1]
        image = Image.fromarray(rgb)
        image.thumbnail((max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height())))
        self._photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self._photo, text='')

    def _sync_model_metadata(self):
        # A source switch can invalidate queued model events after the worker
        # has installed its model. Presentation follows that persistent state.
        model = self.session.recognizer
        if model is self._metadata_model:
            return
        if model is None:
            self.model_var.set('模型：未加载')
            self.vocabulary_label.configure(text='支持词汇：尚未加载模型')
            self._metadata_model = None
            return
        bundle = getattr(model, 'bundle', None)
        if bundle is None:
            return
        manifest = bundle.manifest
        self.model_var.set(f'模型：{manifest.run_id} · {manifest.input_spec.representation_id} · clip')
        self.vocabulary_label.configure(text='支持词汇：' + '、'.join(
            f"{text.get('pt-BR', label)} ({label})" for label, text in bundle.labels.items()))
        self._metadata_model = model

    def _poll(self):
        if self._closed:
            return
        for event in self.session.poll():
            if (event.request_id, event.epoch) != (self.session.request_id, self.session.epoch):
                continue
            try:
                if event.kind == 'clear':
                    self._clear_caption()
                elif event.kind == 'error':
                    self._error(event.value)
                    self.preview.configure(image='', text='来源或任务错误，请重试')
                    self._photo = None
                elif event.kind == 'eof':
                    self.status_var.set(event.value)
                elif event.kind == 'opened':
                    self.status_var.set('来源已打开；' + self._state_text())
                elif event.kind == 'source_closed':
                    if self.session.state not in {'ERROR', 'RESULT', 'REJECTED'} and not self.session.busy:
                        self.status_var.set(event.value)
                elif event.kind == 'result':
                    self._display_result(event.value, event.request_id, event.epoch)
                elif event.kind == 'model':
                    self._sync_model_metadata()
                    self.status_var.set('模型已加载，可识别完整文件或录制动作')
            except Exception as error:
                self._error(error)
        try:
            self._sync_model_metadata()
        except Exception as error:
            self._error(error)
        packet = self.session.take_preview()
        if packet is not None:
            try:
                self._render_preview(packet)
            except Exception as error:
                self._error(error)
        self._update_controls()
        self._after_id = self.root.after(30, self._poll)

    def on_close(self):
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.session.close()
        self.root.destroy()

DesktopApp = SignLanguageApp
