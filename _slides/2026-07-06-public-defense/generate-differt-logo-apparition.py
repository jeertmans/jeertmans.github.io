# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "librosa>=0.11.0",
#     "manim>=0.20.1",
#     "numpy>=2.4.6",
#     "scipy>=1.17.1",
#     "soundfile>=0.14.0",
# ]
# ///
# Run: uv run --with-requirements generate-differt-logo-apparition.py python -m manim render generate-differt-logo-apparition.py -qh -t
import tempfile

import librosa
import manim as m
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt


def apply_bandpass_filter(
    data: np.ndarray,
    sr: float,
    lowcut: float | None,
    highcut: float | None,
    order: int = 5,
) -> np.ndarray:
    if lowcut is None:
        b, a = butter(order, highcut, btype="lowpass", fs=sr)
    elif highcut is None:
        b, a = butter(order, lowcut, btype="highpass", fs=sr)
    else:
        b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=sr)
    return filtfilt(b, a, data)


class AudioReactiveNeon(m.Scene):
    """Audio-reactive scene that animates the DiffeRT logo to music."""

    def construct(self) -> None:
        self.camera.background_opacity = 0

        # --- CONFIGURATION ---
        t_start = 110.0  # Start time in seconds (None for beginning)
        t_end = None  # End time in seconds (None for end)
        t_start_ghost = 10.0  # Time offset after crop start when ghosts begin to appear

        audio_file = "sounds/M83-Solitude.mp3"
        neon_color = "#00ffff"
        subtitle_font = "LIBRARY 3 AM"
        subtitle_text = "Differentiable Ray Tracing for Radio Propagation Modeling"

        # Animation durations
        write_duration = 10.0
        fade_duration = 10.0
        t_subtitle_start = 23.0  # Absolute start time of subtitle typing in seconds
        subtitle_write_duration = 3.0

        # --- AUDIO LOADING & PROCESSING ---
        y_full, sr = librosa.load(audio_file)
        full_duration = librosa.get_duration(y=y_full, sr=sr)
        start_time = 0.0 if t_start is None else max(0.0, t_start)
        end_time = full_duration if t_end is None else min(full_duration, t_end)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = y_full[start_sample:end_sample]
        audio_duration = end_time - start_time

        # Save cropped sound to temporary file for Manim playback
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio:
            sf.write(tmp_audio.name, y, sr)
            self.add_sound(tmp_audio.name)

        # STFT & Frequency Band Extraction
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

        def get_audio_data(t: float) -> tuple[float, float]:
            """Interpolates the bass and high energy normalized values at a given time."""
            current_bass = float(np.interp(t, times, bass_norm))
            current_high = float(np.interp(t, times, high_norm))
            return current_bass, current_high

        # Calibrate Onset Detection
        # Lowpass filter the audio to isolate clean bass transients
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
        logo = m.SVGMobject("images/differt-logo.svg").move_to(m.ORIGIN).scale(2)

        # Create a "Ghost Template" to avoid re-reading SVG from disk on every beat
        ghost_template = logo.copy()
        for subpath in ghost_template:
            subpath.set_style(fill_opacity=0, stroke_width=4, stroke_color=neon_color)

        glow_group = m.VGroup()
        base_aura_opacities = [0.8, 0.5, 0.2]

        for subpath in logo:
            core_line = subpath.copy().set_style(
                fill_opacity=0, stroke_width=2, stroke_color=m.WHITE
            )
            aura_layers = m.VGroup(
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
            glow_group.add(m.VGroup(aura_layers, core_line))

        # Principal fill starts with 0 opacity
        principal_fill = logo.copy()
        for subpath in principal_fill:
            subpath.set_stroke(opacity=0)
            subpath.set_fill(opacity=0)

        # Static clean copy for progressive drawing
        clean_glow_group = glow_group.copy()
        del logo

        # --- SUBTITLE SETUP ---
        subtitle = m.Text(
            subtitle_text,
            font_size=24,
            fill_opacity=0,
            color=neon_color,
            font=subtitle_font,
        )
        subtitle.next_to(glow_group, m.DOWN, buff=0.8)

        # Save reference positions for wiggling
        logo_ref_pos = glow_group.get_center().copy()
        subtitle_ref_pos = subtitle.get_center().copy()

        # --- ANIMATION SEQUENCE ---
        wait_duration = max(0.0, t_subtitle_start - (write_duration + fade_duration))

        def logo_reactive_updater(mobject: m.Mobject, dt: float) -> None:
            t_play = self.time

            # Reset to reference positions before applying new frame's wiggle
            mobject.move_to(logo_ref_pos)
            principal_fill.move_to(logo_ref_pos)

            # 1. Update the Main Logo Opacity and Aura
            bass, high = get_audio_data(t_play)
            core_opacity = np.clip(0.1 + (bass * 0.9), 0.1, 1.0)
            aura_multiplier = np.clip(0.2 + ((high**2) * 1.5), 0.2, 1.5)

            # Apply wiggle effect based on audio energy (frequency peaks)
            max_energy = max(bass, high)
            if max_energy > 0.35:
                factor = (max_energy - 0.35) / 0.65
                logo_wiggle = 0.02 * factor
                wiggle_offset_xy = np.random.uniform(-logo_wiggle, logo_wiggle, size=2)
                logo_offset = np.array([wiggle_offset_xy[0], wiggle_offset_xy[1], 0.0])
                mobject.shift(logo_offset)
                principal_fill.shift(logo_offset)

            # Update progressive drawing (Create effect)
            half_duration = write_duration / 2.0

            # Left part progress (indices 1 and 2)
            prog_left = m.smooth(np.clip(t_play / half_duration, 0.0, 1.0))

            # Right part progress (indices 0 and 3)
            prog_right = m.smooth(
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
                peak_times.pop(0)

                # Clone the template
                ghost = ghost_template.copy()
                ghost.spawn_time = t_play
                ghost.life_time = 0.75  # Ghost lasts for 0.75 seconds
                ghost.current_prog = 0.0

                def ghost_anim(mob: m.Mobject, dt: float) -> None:
                    del dt  # Manim requires a 'dt' argument for updaters to be called on every frame.
                    age = self.time - mob.spawn_time

                    # If too old, remove the ghost and clear its updaters
                    if age > mob.life_time:
                        mob.clear_updaters()
                        self.remove(mob)
                        return

                    # Progress (0.0 to 1.0) with ease-out curve
                    raw_prog = age / mob.life_time
                    prog = 1 - (1 - raw_prog) ** 2

                    # Calculate relative scale growth (total 50% scale increase)
                    scale_ratio = (1.0 + prog * 0.5) / (1.0 + mob.current_prog * 0.5)
                    mob.scale(scale_ratio)

                    # Fade out stroke opacity while keeping fill opacity at 0
                    for sub in mob:
                        sub.set_stroke(opacity=0.5 * (1 - raw_prog))
                        sub.set_fill(opacity=0)

                    mob.current_prog = prog

                ghost.add_updater(ghost_anim)
                self.add(ghost)
                self.bring_to_front(mobject)

        def subtitle_reactive_updater(mobject_group: m.VGroup, dt: float) -> None:
            t_play = self.time
            sub_text_mob, cursor_mob = mobject_group

            if t_play < t_subtitle_start:
                sub_text_mob.set_fill(opacity=0.0)
                cursor_mob.set_opacity(0.0)
                return

            # Reset to reference positions before applying new frame's wiggle
            sub_text_mob.move_to(subtitle_ref_pos)
            cursor_mob.move_to(subtitle_ref_pos)

            # Update subtitle text flickering to match logo energy
            bass, high = get_audio_data(t_play)
            aura_multiplier = np.clip(0.2 + ((high**2) * 1.5), 0.2, 1.5)
            subtitle_opacity = min(
                1.0, 0.3 + (base_aura_opacities[0] * aura_multiplier)
            )

            if t_play > t_subtitle_start + subtitle_write_duration:
                cursor_mob.set_opacity(0.0)
                sub_text_mob.set_fill(opacity=subtitle_opacity)
            else:
                cursor_mob.set_stroke(opacity=1.0)
                idx = int(
                    (t_play - t_subtitle_start)
                    / subtitle_write_duration
                    * len(sub_text_mob)
                )
                idx = min(idx, len(sub_text_mob) - 1)
                cursor_mob.move_to(sub_text_mob[idx])
                sub_text_mob[:idx].set_fill(opacity=subtitle_opacity)

            # Apply wiggle effect based on audio energy
            max_energy = max(bass, high)
            if max_energy > 0.35:
                factor = (max_energy - 0.35) / 0.65
                sub_wiggle = 0.01 * factor
                wiggle_offset_xy = np.random.uniform(-sub_wiggle, sub_wiggle, size=2)
                sub_offset = np.array([wiggle_offset_xy[0], wiggle_offset_xy[1], 0.0])
                sub_text_mob.shift(sub_offset)

        # Add principal fill behind the glow group
        self.add(principal_fill)

        # Attach updaters and add to scene
        glow_group.add_updater(logo_reactive_updater)
        self.add(glow_group)
        self.bring_to_front(glow_group)

        # Setup typewriting cursor
        cursor = m.Rectangle(
            fill_opacity=0.0,
            stroke_color=neon_color,
            stroke_width=1,
            stroke_opacity=0.3,
            height=0.4,
            width=0.2,
        ).move_to(subtitle[0])

        subtitle_group = m.VGroup(subtitle, cursor)
        subtitle_group.add_updater(subtitle_reactive_updater)
        self.add(subtitle_group)

        # Play animations
        self.wait(write_duration)
        self.play(m.FadeIn(principal_fill), run_time=fade_duration)
        self.wait(wait_duration)
        self.wait(max(0.0, audio_duration - self.time))
