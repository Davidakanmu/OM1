import os
import re
import sys
import time
import threading
import logging
from typing import List

import sdl2
import sdl2.ext
import OpenGL.GL as gl
from PIL import Image, ImageDraw, ImageFont

from llm.output_model import Command
from providers.io_provider import IOProvider


class RacoonSim:
    def __init__(self):
        self.messages: List[str] = []
        self.io_provider = IOProvider()
        self.io_provider.fuser_end_time = 0
        self.io_provider.llm_start_time = 0
        self.io_provider.llm_end_time = 0

        # Action state (animation, speech, etc.)
        self.a_s = "idle"  # current action
        self.last_speech = ""
        self.current_emotion = ""
        self.eth_balance = "0.000 ETH"

        # Window dimensions (set as in your pygame version)
        self.X = 1024
        self.Y = 768

        # Colors (RGB 0–255) and background clear color for OpenGL
        self.colors = {
            "bg": (240, 244, 248),
            "panel": (255, 255, 255),
            "primary": (41, 128, 185),
            "accent": (230, 126, 34),
            "text": (44, 62, 80),
            "text_light": (127, 140, 141),
            "debug": (255, 0, 0),
            "success": (46, 204, 113),
            "warning": (241, 196, 15),
            "info": (52, 152, 219),
        }
        bg = self.colors["bg"]
        self.clear_color = (bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

        # Load fonts using PIL. We create a “body” and a “title” font.
        try:
            self.font = ImageFont.truetype("freesansbold.ttf", 16)
            self.title_font = ImageFont.truetype("freesansbold.ttf", 24)
        except IOError:
            self.font = ImageFont.load_default()
            self.title_font = ImageFont.load_default()

        # Load logo (if available) from your assets folder.
        self.path = os.path.join(os.path.dirname(__file__), "assets")
        self.logo_texture = None  # will be created after GL context exists
        logo_path = os.path.join(self.path, "openmind_logo.png")
        if os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                # Resize logo as in your pygame version (100x30)
                self.logo_image = logo_img.resize((100, 30), Image.ANTIALIAS)
            except Exception as e:
                logging.error(f"Error loading logo: {e}")
                self.logo_image = None
        else:
            self.logo_image = None

        # Animation handling
        self.animations = {}  # will be loaded after GL context exists
        self.current_animation = None
        self.animation_frame = 0
        self.last_frame_time = time.time()

        # FPS/timing info (for sidebar)
        self.frame_count = 0
        self.fps_update_time = time.time()
        self.stats = {"fps": 0}
        self.last_frame = time.time()

        # Text textures (for our “main area” and “sidebar”)
        self.main_texture = None
        self.sidebar_texture = None
        self.texture_lock = threading.Lock()

        # Running flag for our OpenGL thread
        self.running = True

        # Start the SDL2/OpenGL lifecycle in a background thread.
        self.thread = threading.Thread(target=self._opengl_loop, daemon=True)
        self.thread.start()

    def load_animations(self):
        """
        Load all GIF animations as (frames, frame_durations) tuples.
        Each frame is loaded as an OpenGL texture.
        """
        def load_gif_frames(filename):
            image_path = os.path.join(self.path, filename)
            try:
                image = Image.open(image_path)
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                return ([], [])
            frames = []
            durations = []
            try:
                while True:
                    img_rgba = image.convert("RGBA")
                    img_data = img_rgba.tobytes("raw", "RGBA")
                    texture_id = gl.glGenTextures(1)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
                    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                    gl.glTexImage2D(
                        gl.GL_TEXTURE_2D,
                        0,
                        gl.GL_RGBA,
                        img_rgba.width,
                        img_rgba.height,
                        0,
                        gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        img_data
                    )
                    frames.append(texture_id)
                    delay = image.info.get("duration", 100) / 1000.0
                    durations.append(delay)
                    image.seek(image.tell() + 1)
            except EOFError:
                pass
            return (frames, durations)

        anim_files = {
            "idle": "idle.gif",
            "sit": "ko.gif",
            "walk": "walk.gif",
            "walk back": "walk_back.gif",
            "run": "run.gif",
            "shake paw": "crouch.gif",
            "dance": "dance.gif",
            "jump": "jump.gif",
        }
        animations = {}
        for action, filename in anim_files.items():
            animations[action] = load_gif_frames(filename)
        return animations

    def _create_texture_from_pil(self, pil_image):
        """Converts a PIL image into an OpenGL texture."""
        pil_image = pil_image.convert("RGBA")
        image_data = pil_image.tobytes("raw", "RGBA")
        width, height = pil_image.size
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            image_data
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return texture_id

    def _opengl_loop(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            logging.error(f"SDL_Init Error: {sdl2.SDL_GetError()}")
            return

        self.window = sdl2.SDL_CreateWindow(
            b"Racoon AI Assistant (SDL2 + OpenGL)",
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            self.X,
            self.Y,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE | sdl2.SDL_WINDOW_SHOWN
        )
        if not self.window:
            logging.error(f"SDL_CreateWindow Error: {sdl2.SDL_GetError()}")
            return

        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        sdl2.SDL_GL_MakeCurrent(self.window, self.gl_context)
        sdl2.SDL_GL_SetSwapInterval(1)

        gl.glViewport(0, 0, self.X, self.Y)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.X, self.Y, 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_TEXTURE_2D)

        # Now that we have a valid GL context, load animations and (if available) the logo texture.
        self.animations = self.load_animations()
        self.current_animation = self.animations.get("idle", ([], []))
        if self.logo_image:
            self.logo_texture = self._create_texture_from_pil(self.logo_image)

        # Main render loop (~60 FPS)
        while self.running:
            self._handle_events()
            self._tick()
            sdl2.SDL_GL_SwapWindow(self.window)
            time.sleep(1 / 60.0)

        sdl2.SDL_GL_DeleteContext(self.gl_context)
        sdl2.SDL_DestroyWindow(self.window)
        sdl2.SDL_Quit()

    def _handle_events(self):
        """Poll SDL events and handle window-resize/quit."""
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                self.running = False
            elif event.type == sdl2.SDL_WINDOWEVENT:
                if event.window.event == sdl2.SDL_WINDOWEVENT_RESIZED:
                    self.X = event.window.data1
                    self.Y = event.window.data2
                    gl.glViewport(0, 0, self.X, self.Y)
                    gl.glMatrixMode(gl.GL_PROJECTION)
                    gl.glLoadIdentity()
                    gl.glOrtho(0, self.X, self.Y, 0, -1, 1)
                    gl.glMatrixMode(gl.GL_MODELVIEW)
                    gl.glLoadIdentity()

    def _draw_filled_rect_gl(self, x, y, w, h, color):
        """Draw a filled rectangle using immediate mode."""
        r, g, b = color
        gl.glColor3f(r / 255.0, g / 255.0, b / 255.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x, y)
        gl.glVertex2f(x + w, y)
        gl.glVertex2f(x + w, y + h)
        gl.glVertex2f(x, y + h)
        gl.glEnd()
        gl.glColor3f(1, 1, 1)

    def _render_texture_gl(self, texture, x, y, w, h):
        """Render an OpenGL texture as a quad at pixel coordinates (x,y) with width w and height h."""
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(x, y)
        gl.glTexCoord2f(1, 0); gl.glVertex2f(x + w, y)
        gl.glTexCoord2f(1, 1); gl.glVertex2f(x + w, y + h)
        gl.glTexCoord2f(0, 1); gl.glVertex2f(x, y + h)
        gl.glEnd()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _render_main_area(self):
        """
        Compose the “main area” as a PIL image.
        This area shows input history (on the left), the animation panel (center),
        and a footer with the logo and version info.
        """
        main_width = self.X - 300  # leaving 300px for the sidebar
        main_height = self.Y
        main_img = Image.new("RGBA", (main_width, main_height), self.colors["bg"] + (255,))
        draw = ImageDraw.Draw(main_img)

        # --- Input History Panel ---
        draw.text((20, 20), "Input History", font=self.title_font, fill=self.colors["primary"])
        history_panel_rect = (20, 55, 220, main_height - 25)
        draw.rectangle(history_panel_rect, fill=self.colors["panel"])
        y_text = 65
        earliest_time = self.get_earliest_time()
        for action, values in self.io_provider.inputs.items():
            line = f"{(values.timestamp - earliest_time):.3f}s :: {action} :: {values.input}"
            draw.text((30, y_text), line, font=self.font, fill=self.colors["text"])
            y_text += 20

        # --- Animation Area Panel ---
        history_width = 200
        available_width = main_width - history_width - 60
        animation_width = 400
        animation_x = history_width + 40 + (available_width - animation_width) // 2
        animation_y = 20
        anim_panel_rect = (animation_x, animation_y, animation_x + animation_width, animation_y + animation_width)
        draw.rectangle(anim_panel_rect, fill=self.colors["panel"])

        # --- Action Label (below the animation panel) ---
        label_y = animation_y + animation_width + 10
        label_rect = (animation_x, label_y, animation_x + animation_width, label_y + 40)
        draw.rectangle(label_rect, fill=self.colors["primary"])
        action_text = f"Current Action: {self.a_s.title()}"
        bbox = draw.textbbox((0, 0), action_text, font=self.font)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        text_x = animation_x + (animation_width - w_text) // 2
        text_y = label_y + (40 - h_text) // 2
        draw.text((text_x, text_y), action_text, font=self.font, fill=self.colors["panel"])

        # --- Footer Panel ---
        footer_height = 40
        footer_y = main_height - footer_height - 10
        footer_rect = (0, footer_y, main_width, footer_y + footer_height)
        draw.rectangle(footer_rect, fill=self.colors["panel"])
        if self.logo_image:
            # Paste the logo into the footer
            logo = self.logo_image.resize((100, 30), Image.ANTIALIAS)
            main_img.paste(logo, (20, footer_y + (footer_height - 30) // 2))
            text_x = 130
        else:
            text_x = 20
        footer_text = "OpenMind AI Assistant v1.0"
        draw.text((text_x, footer_y + (footer_height - 20) // 2), footer_text, font=self.font, fill=self.colors["text"])

        # Return the main image and the rectangle where the animation will be drawn.
        return main_img, (animation_x, animation_y, animation_width, animation_width)

    def _render_sidebar(self):
        """
        Compose the sidebar (300px wide) as a PIL image.
        The sidebar shows ETH balance, system status (including FPS and timings),
        and a list of available commands.
        """
        sidebar_width = 300
        sidebar_height = self.Y
        sidebar_img = Image.new("RGBA", (sidebar_width, sidebar_height), self.colors["panel"] + (255,))
        draw = ImageDraw.Draw(sidebar_img)

        y = 20
        # --- ETH Balance Panel ---
        balance_rect = (20, y, 280, y + 80)
        draw.rectangle(balance_rect, fill=self.colors["success"])
        draw.text((35, y + 10), "ETH Balance", font=self.title_font, fill=self.colors["panel"])
        draw.text((35, y + 40), self.eth_balance, font=self.title_font, fill=self.colors["panel"])
        y += 100

        # --- System Status ---
        draw.text((20, y), "System Status", font=self.title_font, fill=self.colors["primary"])
        y += 35
        draw.text((20, y), f"FPS: {self.stats['fps']:.1f}", font=self.font, fill=self.colors["text"])
        y += 30
        earliest_time = self.get_earliest_time()
        timing_data = [
            ("Fuse time:", f"{self.io_provider.fuser_end_time - earliest_time:.3f}s"),
            ("LLM start:", f"{float(self.io_provider.llm_start_time) - earliest_time:.3f}s"),
            ("Processing:", f"{(float(self.io_provider.llm_end_time) - float(self.io_provider.llm_start_time)):.3f}s"),
            ("Complete:", f"{float(self.io_provider.llm_end_time) - earliest_time:.3f}s"),
        ]
        for label, value in timing_data:
            draw.text((20, y), label, font=self.font, fill=self.colors["text_light"])
            draw.text((120, y), value, font=self.font, fill=self.colors["text"])
            y += 25

        y += 30
        draw.text((20, y), "Available Commands", font=self.title_font, fill=self.colors["primary"])
        y += 35
        commands = {
            "Movement": [
                "walk - Walk forward",
                "run - Run quickly",
                "jump - Jump up",
                "dance - Do a dance",
                "sit - Sit down",
                "shake paw - Shake paw",
                "walk back - Walk backward",
            ],
            "Expressions": [
                "face: smile - Happy",
                "face: think - Thoughtful",
                "face: frown - Sad",
                "face: cry - Crying",
            ],
            "Speech": [
                "speech - Speak a message",
                "bark - Bark sound",
            ]
        }
        for category, cmds in commands.items():
            draw.text((20, y), category, font=self.font, fill=self.colors["accent"])
            y += 25
            for cmd in cmds:
                draw.text((30, y), f"• {cmd}", font=self.font, fill=self.colors["text"])
                y += 20
            y += 15

        return sidebar_img

    def _tick(self):
        """
        The main update loop. It:
          1. Updates FPS and the current animation frame.
          2. Composes (via PIL) the main area and sidebar images.
          3. Uploads them as OpenGL textures.
          4. Renders the two panels plus the current animation frame.
        """
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.fps_update_time >= 1.0:
            self.stats["fps"] = self.frame_count
            self.frame_count = 0
            self.fps_update_time = current_time

        # Update animation frame
        frames, durations = self.current_animation
        if frames:
            if current_time - self.last_frame_time >= durations[self.animation_frame]:
                self.animation_frame = (self.animation_frame + 1) % len(frames)
                self.last_frame_time = current_time
            current_anim_texture = frames[self.animation_frame]
        else:
            current_anim_texture = None

        # Compose main and sidebar images (using PIL)
        with self.texture_lock:
            main_img, anim_area = self._render_main_area()
            sidebar_img = self._render_sidebar()
            # Delete old textures (if any) and create new ones
            if self.main_texture:
                gl.glDeleteTextures(1, [self.main_texture])
            if self.sidebar_texture:
                gl.glDeleteTextures(1, [self.sidebar_texture])
            self.main_texture = self._create_texture_from_pil(main_img)
            self.sidebar_texture = self._create_texture_from_pil(sidebar_img)

        # Clear the screen
        gl.glClearColor(*self.clear_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Render the main area (left)
        main_width = self.X - 300
        main_height = self.Y
        self._render_texture_gl(self.main_texture, 0, 0, main_width, main_height)
        # Render the sidebar (right)
        self._render_texture_gl(self.sidebar_texture, main_width, 0, 300, self.Y)
        # Render the current animation frame on top of the main area (inside the animation panel)
        if current_anim_texture:
            anim_x, anim_y, anim_w, anim_h = anim_area
            self._render_texture_gl(current_anim_texture, anim_x, anim_y, anim_w, anim_h)

    def input_clean(self, input_str, earliest_time) -> str:
        """Cleans up the input string (as in your pygame code)."""
        st = input_str.strip().replace("\n", "")
        st = re.sub(r"\s+", " ", st)
        st = st.replace("INPUT // START ", "")
        st = st.replace(" // END", "")
        sts = st.split("::")
        time_val = float(sts[0])
        time_rezero = time_val - earliest_time
        return f"{time_rezero:.3f}::{sts[-1]}"

    def get_earliest_time(self) -> float:
        """Returns the earliest timestamp among the inputs."""
        earliest_time = float("inf")
        for value in self.io_provider.inputs.values():
            if value.timestamp < earliest_time:
                earliest_time = value.timestamp
        return earliest_time if earliest_time != float("inf") else 0.0

    def sim(self, commands: List[Command]) -> None:
        """
        Updates the simulation state. In this version we process:
          - move commands (to update the current action)
          - speech, face, and wallet commands (to update status shown in the sidebar)
        """
        earliest_time = self.get_earliest_time()
        for command in commands:
            if command.name == "move":
                new_action = command.arguments[0].value
                self.a_s = new_action
                # Update the current animation if available; reset frame counter.
                self.current_animation = self.animations.get(new_action, self.animations.get("idle"))
                self.animation_frame = 0
                self.last_frame_time = time.time()
            elif command.name == "speech":
                text = command.arguments[0].value
                self.last_speech = text if len(text) <= 50 else text[:50] + "..."
            elif command.name == "face":
                self.current_emotion = command.arguments[0].value
            elif command.name == "wallet":
                try:
                    balance = float(command.arguments[0].value)
                    self.eth_balance = f"{balance:.3f} ETH"
                except Exception as e:
                    logging.error(f"Error updating wallet: {e}")
                    self.eth_balance = "0.000 ETH"

    def tick(self) -> None:
        """
        For drop‐in compatibility. In the pygame version tick() drives the update loop.
        Here, since _tick() is being called in the background thread, we simply sleep.
        """
        time.sleep(1 / 200)

    def stop(self):
        """Stops the rendering loop and waits for the background thread to finish."""
        self.running = False
        self.thread.join()


