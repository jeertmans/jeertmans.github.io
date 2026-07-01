import tempfile

import librosa
import numpy as np
import soundfile as sf
from manim import *
from scipy.signal import butter, filtfilt


def apply_bandpass_filter(data, sr, lowcut, highcut, order=5):
    """
    Applies a zero-phase Butterworth bandpass filter to an audio signal.
    """

    # Generate the Butterworth filter coefficients
    if lowcut is None:
        b, a = butter(order, highcut, btype="lowpass", fs=sr)
    elif highcut is None:
        b, a = butter(order, lowcut, btype="highpass", fs=sr)
    else:
        b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=sr)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


class AudioReactiveNeon(Scene):
    def construct(self):
        self.camera.background_opacity = 0

        t_start = 110.0  # Start time in seconds (set to None for the beginning)
        t_end = None  # End time in seconds (set to None for the end)
        t_start_ghost = (
            10.0  # Ghosts only appear after 10.0 seconds from the crop start
        )

        audio_file = "sounds/M83-Solitude.mp3"

        # 1. Load full audio data
        y_full, sr = librosa.load(audio_file)
        full_duration = librosa.get_duration(y=y_full, sr=sr)
        start_time = 0.0 if t_start is None else max(0.0, t_start)
        end_time = full_duration if t_end is None else min(full_duration, t_end)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = y_full[start_sample:end_sample]
        audio_duration = end_time - start_time

        # Save cropped sound to temporary file and immediately close it after self.add_sound
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio:
            sf.write(tmp_audio.name, y, sr)
            self.add_sound(tmp_audio.name)

        # --- AUDIO PROCESSING (STFT & Onset Detection) ---

        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        bass_idx = np.where((freqs >= 20) & (freqs <= 100))[0]
        high_idx = np.where((freqs >= 2000) & (freqs <= 8000))[0]

        bass_energy = (
            np.mean(D[bass_idx, :], axis=0)
            if len(bass_idx) > 0
            else np.zeros(D.shape[1])
        )
        high_energy = (
            np.mean(D[high_idx, :], axis=0)
            if len(high_idx) > 0
            else np.zeros(D.shape[1])
        )

        bass_norm = bass_energy / (np.max(bass_energy) + 1e-6)
        high_norm = high_energy / (np.max(high_energy) + 1e-6)

        times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)

        def get_audio_data(t):
            current_bass = np.interp(t, times, bass_norm)
            current_high = np.interp(t, times, high_norm)
            return current_bass, current_high

        # 2. CALIBRATE ONSET DETECTION (Migrated from audio-filter.py)
        # Apply lowpass filter to extract only bass frequencies for clean peak detection
        y_bass_filtered = apply_bandpass_filter(y, sr, None, 100)

        onset_env = librosa.onset.onset_strength(
            y=y_bass_filtered, sr=sr, hop_length=512, aggregate=np.mean
        )

        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=10,
            post_max=10,
            pre_avg=10,
            post_avg=10,
            delta=0.15,
            wait=int(0.2 * sr / 512),
        )

        peak_times = sorted(list(librosa.frames_to_time(peaks, sr=sr, hop_length=512)))
        # Only keep peaks that occur after t_start_ghost
        peak_times = [t for t in peak_times if t >= t_start_ghost]

        # --- VISUAL SETUP ---
        logo = SVGMobject("images/differt-logo.svg").move_to(ORIGIN).scale(2)
        neon_color = "#00ffff"

        # Create a "Ghost Template" so we don't have to read the SVG from the hard drive every beat
        ghost_template = logo.copy()
        for subpath in ghost_template:
            subpath.set_style(fill_opacity=0, stroke_width=4, stroke_color=neon_color)

        glow_group = VGroup()
        base_aura_opacities = [0.8, 0.5, 0.2]

        for subpath in logo:
            core_line = subpath.copy().set_style(
                fill_opacity=0, stroke_width=2, stroke_color=WHITE
            )
            aura_layers = VGroup(
                subpath.copy().set_style(
                    fill_opacity=0,
                    stroke_width=4,
                    stroke_color=neon_color,
                    stroke_opacity=base_aura_opacities[0],
                ),
                subpath.copy().set_style(
                    fill_opacity=0,
                    stroke_width=8,
                    stroke_color=neon_color,
                    stroke_opacity=base_aura_opacities[1],
                ),
                subpath.copy().set_style(
                    fill_opacity=0,
                    stroke_width=15,
                    stroke_color=neon_color,
                    stroke_opacity=base_aura_opacities[2],
                ),
            )
            glow_group.add(VGroup(aura_layers, core_line))

        # Keep a copy for the principal fill, with no stroke, and start with fill opacity 0
        principal_fill = logo.copy()
        for subpath in principal_fill:
            subpath.set_stroke(opacity=0)
            subpath.set_fill(opacity=0)

        # Create a clean static copy of the glow group for progressive drawing (Create effect)
        clean_glow_group = glow_group.copy()

        del logo

        # --- SUBTITLE SETUP ---
        subtitle = Text(
            "Differentiable Ray Tracing for Radio Propagation Modeling",
            font_size=24,
            fill_opacity=0,
            color=neon_color,
            font="LIBRARY 3 AM",
        )
        subtitle.next_to(glow_group, DOWN, buff=0.8)

        # Save reference positions for wiggling
        logo_ref_pos = glow_group.get_center().copy()
        subtitle_ref_pos = subtitle.get_center().copy()

        # --- ANIMATION SEQUENCE ---

        # Define duration variables early so they are accessible inside the updater
        write_duration = 10.0
        fade_duration = 10.0
        t_subtitle_start = 23.0  # Absolute start time of subtitle typing in seconds
        subtitle_write_duration = 3.0

        wait_duration = max(0.0, t_subtitle_start - (write_duration + fade_duration))

        glow_group.playback_time = 0.0

        def logo_reactive_updater(mobject, dt: float):
            t_play = self.time

            # Reset to reference positions before applying new frame's wiggle
            mobject.move_to(logo_ref_pos)
            principal_fill.move_to(logo_ref_pos)

            # 1. Update the Main Logo Opacity
            bass, high = get_audio_data(t_play)
            core_opacity = np.clip(0.1 + (bass * 0.9), 0.1, 1.0)
            aura_multiplier = np.clip(0.2 + ((high**2) * 1.5), 0.2, 1.5)

            # Apply wiggle effect based on audio energy (frequency peaks)
            max_energy = max(bass, high)
            if max_energy > 0.35:
                factor = (max_energy - 0.35) / 0.65
                # Logo wiggle (reduced amplitude to 0.02)
                logo_wiggle = 0.02 * factor
                logo_offset = np.random.uniform(-logo_wiggle, logo_wiggle, size=3)
                # logo_offset[2] = 0.0
                mobject.shift(logo_offset)
                principal_fill.shift(logo_offset)

            # 1. Update progressive drawing (Create effect)
            write_duration = 10.0
            half_duration = write_duration / 2.0

            # Left part progress (indices 1 and 2)
            prog_left = smooth(np.clip(t_play / half_duration, 0.0, 1.0))

            # Right part progress (indices 0 and 3)
            prog_right = smooth(
                np.clip((t_play - half_duration) / half_duration, 0.0, 1.0)
            )

            for i, part in enumerate(mobject):
                clean_part = clean_glow_group[i]
                aura_layers = part[0]
                core_line = part[1]

                # Determine which progress to use
                write_prog = prog_left if i in (1, 2) else prog_right

                # Apply partial drawing (Create effect)
                core_line.pointwise_become_partial(clean_part[1], 0, write_prog)
                core_line.set_stroke(opacity=core_opacity)

                # Apply partial drawing and flickering to aura layers
                for j, layer in enumerate(aura_layers):
                    layer.pointwise_become_partial(clean_part[0][j], 0, write_prog)
                    new_opacity = min(1.0, base_aura_opacities[j] * aura_multiplier)
                    layer.set_stroke(opacity=new_opacity)

            # 2. Check for Bass Drops & Spawn Ghosts
            nonlocal peak_times
            while len(peak_times) > 0 and t_play >= peak_times[0]:
                peak_times.pop(0)  # Remove the timestamp we just crossed

                # Clone the template
                ghost = ghost_template.copy()
                ghost.spawn_time = t_play
                ghost.life_time = 0.75  # Ghost lasts for 1 second
                ghost.current_prog = 0.0

                def ghost_anim(mob, dt):
                    age = self.time - mob.spawn_time

                    # If it's too old, delete it to save memory and performance
                    if age > mob.life_time:
                        mob.clear_updaters()
                        self.remove(mob)
                        return

                    # Calculate progress (0.0 to 1.0) with an ease-out curve
                    raw_prog = age / mob.life_time
                    prog = 1 - (1 - raw_prog) ** 2

                    # Calculate relative scaling (so it grows smoothly frame-by-frame)
                    # We want it to grow by 50% overall (1.0 -> 1.5)
                    scale_ratio = (1.0 + prog * 0.5) / (1.0 + mob.current_prog * 0.5)
                    mob.scale(scale_ratio)

                    # Fade out stroke opacity while keeping fill opacity at 0
                    for subpath in mob:
                        subpath.set_stroke(opacity=0.5 * (1 - raw_prog))
                        subpath.set_fill(opacity=0)

                    mob.current_prog = prog

                # Attach the animation logic to the ghost and add it to the scene
                ghost.add_updater(ghost_anim)
                self.add(ghost)

                # Ensure the main logo stays on top of the newly spawned ghost
                self.bring_to_front(mobject)

        def subtitle_reactive_updater(mobject_group, dt: float):
            t_play = self.time

            subtitle, cursor = mobject_group

            if t_play < t_subtitle_start:
                subtitle.set_fill(opacity=0.0)
                cursor.set_opacity(0.0)
                return

            # Reset to reference positions before applying new frame's wiggle
            subtitle.move_to(subtitle_ref_pos)
            cursor.move_to(subtitle_ref_pos)

            # 1. Update the subtitle text flickering to match the logo
            bass, high = get_audio_data(t_play)
            core_opacity = np.clip(0.1 + (bass * 0.9), 0.1, 1.0)
            aura_multiplier = np.clip(0.2 + ((high**2) * 1.5), 0.2, 1.5)
            subtitle_opacity = min(
                1.0, 0.3 + (base_aura_opacities[0] * aura_multiplier)
            )

            if t_play > t_subtitle_start + subtitle_write_duration:
                cursor.set_opacity(0.0)
                subtitle.set_fill(opacity=subtitle_opacity)
            else:
                cursor.set_stroke(opacity=1.0)
                idx = int(
                    (t_play - t_subtitle_start)
                    / subtitle_write_duration
                    * len(subtitle)
                )
                cursor.move_to(subtitle[idx])
                subtitle[:idx].set_fill(opacity=subtitle_opacity)

            # Apply wiggle effect based on audio energy (frequency peaks)
            max_energy = max(bass, high)
            if max_energy > 0.35:
                factor = (max_energy - 0.35) / 0.65
                sub_wiggle = 0.01 * factor
                sub_offset = np.random.uniform(-sub_wiggle, sub_wiggle, size=3)
                # sub_offset[2] = 0.0
                subtitle.shift(sub_offset)

        # Add principal fill behind the glow group (initially 0 opacity)
        self.add(principal_fill)

        # Start updaters on both glow group and subtitle, and add them to the scene
        glow_group.add_updater(logo_reactive_updater)
        self.add(glow_group)
        self.bring_to_front(glow_group)

        cursor = Rectangle(
            fill_opacity=0.0,
            stroke_color=neon_color,
            stroke_width=1,
            stroke_opacity=0.3,
            height=0.4,
            width=0.2,
        ).move_to(subtitle[0])
        subtitle_group = VGroup(subtitle, cursor)
        subtitle_group.add_updater(subtitle_reactive_updater)
        self.add(subtitle_group)

        # 1. Gradual creation of the contour lines (while they blink reactively)
        self.wait(write_duration)

        # 2. Fade in the principal fill after writing is done
        self.play(FadeIn(principal_fill), run_time=fade_duration)

        # 3. Wait reactive until near the end of audio duration
        self.wait(wait_duration)

        # 4. Write subtitle at the end
        self.wait(max(0.0, audio_duration - self.time))
