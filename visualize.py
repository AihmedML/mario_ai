import cv2
import numpy
import torch


class Visualizer:
    def __init__(self, action_names):
        self.action_names = action_names
        self.reward_history = []
        self.best_reward = float("-inf")

    def show(self, state, agent, episode, total_reward):
        panel = numpy.zeros((700, 1000, 3), dtype=numpy.uint8)

        # --- top left: what AI sees ---
        frame = state[0]
        frame_big = cv2.resize(frame, (200, 200))
        frame_rgb = numpy.stack([frame_big * 255] * 3, axis=-1).astype(numpy.uint8)
        panel[30:230, 10:210] = frame_rgb
        cv2.putText(
            panel, "AI Vision", (55, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1
        )

        # --- get policy probs and value (skip activations for speed) ---
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            aux_tensor = torch.tensor(agent.current_aux[0:1], dtype=torch.float32).to(agent.device)
            logits, value = agent.net(state_tensor, aux_tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            state_value = value.item()
        activations = []

        # --- top left below frame: action probability bars ---
        best_action = probs.argmax()

        cv2.putText(
            panel, "Action Probabilities", (30, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1
        )
        for i, (name, p) in enumerate(zip(self.action_names, probs)):
            y = 270 + i * 32
            bar_width = int(p * 250)

            if i == best_action:
                color = (0, 255, 0)
            else:
                color = (0, 120, 0)

            cv2.rectangle(panel, (160, y), (160 + bar_width, y + 22), color, -1)
            cv2.putText(
                panel, name, (10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            cv2.putText(
                panel, f"{p:.0%}", (330, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1
            )

        # --- middle: NN layer activations ---
        cv2.putText(
            panel, "Network Activations", (440, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1
        )

        y_offset = 35

        if activations:
            layer_names = ["Conv1 (32 filters)", "Conv2 (64 filters)", "Conv3 (64 filters)"]
            for layer_idx, (act, lname) in enumerate(zip(activations, layer_names)):
                cv2.putText(
                    panel, lname, (420, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1
                )
                y_offset += 18

                n_show = min(8, act.shape[0])
                size = 45

                for i in range(n_show):
                    row, col = i // 4, i % 4
                    filt = act[i]
                    filt_min, filt_max = filt.min(), filt.max()
                    if filt_max - filt_min > 0:
                        filt = ((filt - filt_min) / (filt_max - filt_min) * 255).astype(numpy.uint8)
                    else:
                        filt = numpy.zeros_like(filt, dtype=numpy.uint8)

                    filt_resized = cv2.resize(filt, (size, size))
                    filt_color = cv2.applyColorMap(filt_resized, cv2.COLORMAP_VIRIDIS)

                    px = 420 + col * (size + 3)
                    py = y_offset + row * (size + 3)

                    panel[py : py + size, px : px + size] = filt_color

                y_offset += 2 * (size + 3) + 10
        else:
            cv2.putText(
                panel, "(Disabled for speed)", (440, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1
            )
            y_offset = 80

        # --- NN architecture diagram ---
        diag_y = y_offset + 20
        cv2.putText(
            panel, "Architecture (PPO Actor-Critic)", (420, diag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1
        )
        diag_y += 15

        layers = ["4x84x84", "32x26x26", "64x12x12", "64x10x10", "512", "P:7 V:1"]
        box_w = 70
        box_h = 25
        start_x = 420
        cols = 3

        for i, label in enumerate(layers):
            row, col = i // cols, i % cols
            x = start_x + col * (box_w + 12)
            y = diag_y + row * (box_h + 15)
            brightness = 80 + int((i / len(layers)) * 175)
            color = (brightness, 50, 0)
            cv2.rectangle(panel, (x, y), (x + box_w, y + box_h), color, -1)
            cv2.rectangle(panel, (x, y), (x + box_w, y + box_h), (150, 150, 150), 1)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = x + (box_w - text_size[0]) // 2
            cv2.putText(
                panel, label, (text_x, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
            )

            if i < len(layers) - 1 and col < cols - 1:
                cv2.arrowedLine(
                    panel,
                    (x + box_w, y + box_h // 2),
                    (x + box_w + 10, y + box_h // 2),
                    (100, 100, 100),
                    1,
                    tipLength=0.4,
                )

        # --- right: training stats ---
        stats_x = 750
        cv2.putText(
            panel, "Training Stats", (stats_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1
        )
        lr = agent.optimizer.param_groups[0]["lr"]
        stats = [
            (f"Episode: {episode}", (255, 255, 255)),
            (f"Reward: {total_reward:.0f}", (255, 255, 255)),
            (f"Best: {self.best_reward:.0f}", (0, 255, 0)),
            (f"Value: {state_value:.2f}", (0, 255, 255)),
            (f"Steps: {agent.step_count}", (255, 255, 255)),
            (f"LR: {lr:.6f}", (255, 200, 0)),
            (f"Envs: {agent.num_envs} | {agent.device}", (100, 100, 255)),
        ]
        for i, (text, color) in enumerate(stats):
            cv2.putText(
                panel, text, (stats_x, 50 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # --- bottom right: reward history graph ---
        graph_x, graph_y = 700, 280
        graph_w, graph_h = 280, 200

        cv2.putText(
            panel, "Reward History", (graph_x, graph_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1
        )
        cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (40, 40, 40), -1)
        cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (80, 80, 80), 1)

        if len(self.reward_history) > 1:
            rewards = self.reward_history
            max_r = max(rewards)
            min_r = min(rewards)
            r_range = max_r - min_r if max_r != min_r else 1

            # axis labels
            cv2.putText(
                panel, f"{max_r:.0f}", (graph_x + 2, graph_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1
            )
            cv2.putText(
                panel, f"{min_r:.0f}", (graph_x + 2, graph_y + graph_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1
            )

            n = len(rewards)
            points = []
            for i in range(n):
                x = graph_x + int(i / max(n - 1, 1) * (graph_w - 10)) + 5
                y = graph_y + graph_h - 5 - int((rewards[i] - min_r) / r_range * (graph_h - 15))
                points.append((x, y))

            # raw reward line
            for i in range(1, len(points)):
                cv2.line(panel, points[i - 1], points[i], (0, 80, 0), 1)

            # moving average line
            if n >= 10:
                window = min(20, n // 2)
                avg = []
                for i in range(n):
                    start = max(0, i - window + 1)
                    avg.append(sum(rewards[start : i + 1]) / (i - start + 1))

                avg_points = []
                for i in range(n):
                    x = graph_x + int(i / max(n - 1, 1) * (graph_w - 10)) + 5
                    y = graph_y + graph_h - 5 - int((avg[i] - min_r) / r_range * (graph_h - 15))
                    avg_points.append((x, y))

                for i in range(1, len(avg_points)):
                    cv2.line(panel, avg_points[i - 1], avg_points[i], (0, 255, 0), 2)

            # legend
            cv2.line(panel, (graph_x + 10, graph_y + graph_h + 15), (graph_x + 30, graph_y + graph_h + 15), (0, 80, 0), 1)
            cv2.putText(panel, "Raw", (graph_x + 35, graph_y + graph_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            cv2.line(panel, (graph_x + 70, graph_y + graph_h + 15), (graph_x + 90, graph_y + graph_h + 15), (0, 255, 0), 2)
            cv2.putText(panel, "Avg", (graph_x + 95, graph_y + graph_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        else:
            cv2.putText(
                panel, "Waiting for data...", (graph_x + 60, graph_y + graph_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1
            )

        # --- bottom: entropy bar (exploration indicator for PPO) ---
        bar_y = 670
        bar_x = 10
        bar_w = 680
        entropy = -numpy.sum(probs * numpy.log(probs + 1e-8))
        max_entropy = numpy.log(len(self.action_names))
        entropy_ratio = entropy / max_entropy

        cv2.putText(
            panel, "Low Entropy (Confident)", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1
        )
        cv2.putText(
            panel, "High Entropy (Exploring)", (bar_x + bar_w - 160, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1
        )
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (40, 40, 40), -1)
        fill_w = int(entropy_ratio * bar_w)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + 15), (0, 180, 180), -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (80, 80, 80), 1)

        cv2.imshow("AI Brain", panel)
        cv2.waitKey(1)

    def end_episode(self, total_reward):
        self.reward_history.append(total_reward)
        if total_reward > self.best_reward:
            self.best_reward = total_reward
