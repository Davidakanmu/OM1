import os
import re
import time
from typing import List

import gif_pygame
import pygame

#import logging
from llm.output_model import Command
from providers.io_provider import IOProvider


class RacoonSim:
    def __init__(self):
        self.messages: list[str] = []

        self.io_provider = IOProvider()

        self.name = __class__
        pygame.init()

        # define the RGB value for white,
        #  green, blue colour .
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 128)
        self.black = (0, 0, 0)
        self.lightblue = (84, 118, 156)

        self.X = 800
        self.Y = 630

        self.display = pygame.display.set_mode((self.X, self.Y))
        pygame.display.set_caption("Racoon Simulator")
        self.display.fill(self.lightblue)

        self.surface_text = pygame.Surface((800, 230))
        self.surface_text.fill(self.white)

        self.surface_ani = pygame.Surface((400, 400))
        self.surface_ani.fill(self.lightblue)

        self.font = pygame.font.Font("freesansbold.ttf", 14)
        self.path = os.path.join(os.path.dirname(__file__), "assets")

        self.action_idle = gif_pygame.load(os.path.join(self.path, "idle.gif"))
        self.action_sit = gif_pygame.load(os.path.join(self.path, "ko.gif"))
        self.action_walk = gif_pygame.load(os.path.join(self.path, "walk.gif"))
        self.action_walk_back = gif_pygame.load(os.path.join(self.path, "walk_back.gif"))
        self.action_run = gif_pygame.load(os.path.join(self.path, "run.gif"))
        self.action_shake_paw = gif_pygame.load(os.path.join(self.path, "crouch.gif"))
        self.action_dance = gif_pygame.load(os.path.join(self.path, "dance.gif"))
        self.action_jump = gif_pygame.load(os.path.join(self.path, "jump.gif"))

        self.a_s = ""

    def tick(self) -> None:
        time.sleep(1 / 200)
        self._tick()

    def _tick(self) -> None:
        self.surface_ani.fill(self.lightblue)

        #logging.info(f"Current Action Spec: {self.a_s}")

        if self.a_s == "walk":
            #logging.info(f"Current Action Spec: {self.a_s} - using WALK render")
            self.action_walk.render(self.surface_ani, (0, 0))
        elif self.a_s == "walk back":
            #logging.info(f"Current Action Spec: {self.a_s} - using WALK BACK render")
            self.action_walk_back.render(self.surface_ani, (0, 0))
        elif self.a_s == "run":
            #logging.info(f"Current Action Spec: {self.a_s} - using RUN render")
            self.action_run.render(self.surface_ani, (0, 0))
        elif self.a_s == "sit":
            #logging.info(f"Current Action Spec: {self.a_s} - using SIT render")
            self.action_sit.render(self.surface_ani, (0, 0))
        elif self.a_s == "shake paw":
            #logging.info(f"Current Action Spec: {self.a_s} - using SHAKE PAW render")
            self.action_shake_paw.render(self.surface_ani, (0, 0))
        elif self.a_s == "dance":
            #logging.info(f"Current Action Spec: {self.a_s} - using DANCE render")
            self.action_dance.render(self.surface_ani, (0, 0))
        elif self.a_s == "stand still":
            #logging.info(f"Current Action Spec: {self.a_s} - using IDLE render")
            self.action_idle.render(self.surface_ani, (0, 0))
        elif self.a_s == "jump":
            #logging.info(f"Current Action Spec: {self.a_s} - using JUMP render")
            self.action_jump.render(self.surface_ani, (0, 0))
        else:
            # logging.info(f"Could not parse action spec {self.a_s} - defaulting to idle")
            self.action_idle.render(self.surface_ani, (0, 0))

        self.display.blit(self.surface_ani, (180, 230))

        # this is what updates everything
        pygame.display.flip()

    def input_clean(self, input, earliest_time) -> str:
        st = input
        st = st.strip()
        st = st.replace("\n", "")
        st = re.sub(r"\s+", " ", st)  # replace runs of whitespace
        st = st.replace("INPUT // START ", "")
        st = st.replace(" // END", "")

        sts = st.split("::")
        time = float(sts[0])
        time_rezero = time - earliest_time
        time_st = f"{time_rezero:.3f}::{sts[-1]}"

        return time_st

    def get_earliest_time(self) -> float:
        earliest_time = float("inf")
        for value in self.io_provider.inputs.values():
            timestamp = value.timestamp
            if timestamp < earliest_time:
                earliest_time = timestamp
        return earliest_time

    def sim(self, commands: List[Command]) -> None:
        earliest_time = self.get_earliest_time()

        # make the background white
        self.surface_text.fill(self.white)

        y = 15
        for action, values in self.io_provider.inputs.items():
            inp = (
                f"{(values.timestamp - earliest_time):.3f}:: {action} :: {values.input}"
            )
            self.text = self.font.render(inp, True, self.black, self.white)
            self.textRect = self.text.get_rect()
            self.textRect.topleft = (20, y)
            y += 20
            self.surface_text.blit(self.text, self.textRect)

        dft = f"{self.io_provider.fuser_end_time - earliest_time:.3f}"
        self.text = self.font.render(f"Fuse_time: {dft}", True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (20, y)
        y += 20
        self.surface_text.blit(self.text, self.textRect)

        dst = f"{float(self.io_provider.llm_start_time) - earliest_time:.3f}"
        self.text = self.font.render(f"LLM_begin: {dst}", True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (20, y)
        y += 20
        self.surface_text.blit(self.text, self.textRect)

        processing_time_s = float(self.io_provider.llm_end_time) - float(
            self.io_provider.llm_start_time
        )
        dt = f"{processing_time_s:.3f}"
        self.text = self.font.render(
            f"LLM_proc time: {dt}", True, self.black, self.white
        )
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (20, y)
        y += 20
        self.surface_text.blit(self.text, self.textRect)

        dtt = f"{float(self.io_provider.llm_end_time) - earliest_time:.3f}"
        self.text = self.font.render(f"LLM_done: {dtt}", True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (20, y)
        y += 20
        self.surface_text.blit(self.text, self.textRect)

        for command in commands:
            action_type = command.name
            action_spec = command.arguments[0].value
            if action_type == "move":
                self.a_s = action_spec
            action = action_type + "::" + action_spec
            self.text = self.font.render(action, True, self.black, self.white)
            self.textRect = self.text.get_rect()
            self.textRect.topleft = (20, y)
            y += 20
            self.surface_text.blit(self.text, self.textRect)

        self.display.blit(self.surface_text, (0, 0))
