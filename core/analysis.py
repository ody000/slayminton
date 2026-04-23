'''
Analyses for shuttle and player tracking data. Implemented in week 2.
Gives a higher level intuition of the data.
'''

import os

import matplotlib.pyplot as plt

class Analysis:
    def __init__(self):
        # Keep this class focused on rally duration stats for now.
        self.last_results = None

    def analyze_shuttle_trajectories(self, shuttle_detections):
        # Placeholder for trajectory analysis logic
        pass

    def compute_rally_statistics(self, rally_data):
        """
        Compute statstics for each rally. Key metrics:
        - Rally duration

        Later extensions:
        - Number of hits
        - Average shuttle speed
        - Player movement distance
        - Average time between hits

        Rally data is currently recorded in JSON output from tracking loop
        modified by GameState updates.
        """
        # Expected input format per rally item:
        # {"rally_id": int, "start_time": float, "end_time": float, "duration_s": float}
        normalized = []
        for idx, item in enumerate(rally_data or []):
            if item is None:
                continue

            # Be tolerant to missing duration_s by deriving from start/end.
            start_time = item.get("start_time", None)
            end_time = item.get("end_time", None)
            duration_s = item.get("duration_s", None)

            if duration_s is None and start_time is not None and end_time is not None:
                duration_s = float(end_time) - float(start_time)

            if duration_s is None:
                continue

            duration_s = max(float(duration_s), 0.0)
            normalized.append(
                {
                    "rally_id": int(item.get("rally_id", idx + 1)),
                    "start_time": None if start_time is None else float(start_time),
                    "end_time": None if end_time is None else float(end_time),
                    "duration_s": duration_s,
                }
            )

        durations = [r["duration_s"] for r in normalized]
        rally_count = len(durations)

        results = {
            "rallies": normalized,
            "rally_count": rally_count,
            "total_rally_duration_s": float(sum(durations)) if durations else 0.0,
            "mean_rally_duration_s": float(sum(durations) / rally_count) if rally_count else 0.0,
            "min_rally_duration_s": float(min(durations)) if durations else 0.0,
            "max_rally_duration_s": float(max(durations)) if durations else 0.0,
            "durations_s": durations,
        }

        self.last_results = results
        return results

    def analyze_player_movements(self, player_detections):
        """
        Process player detections to extract movement patterns. Key steps:
        - Smooth trajectories to reduce noise
        - Compute movement vectors and speeds
        - Compute and visualize heatmaps of player positions on the court

        Later potential extensions:
        - Identify common movement patterns (e.g. cross-court, net play)
        - Correlate player movement with shuttle trajectories (for identifying strategies)
        - Compare player movement patterns across rallies
        """
        pass

    def visualize_results(self, analysis_results):
        """
        Using matplotlib.
        Key visualizations:
        - the distribution of rally durations (histogram)

        Later extensions:
        - heatmaps of player positions on the court
        - hit positions of shuttle landings on the court
        """
        # Accept either a results dict or (results, output_dir) tuple.
        if isinstance(analysis_results, tuple):
            stats = analysis_results[0] if len(analysis_results) > 0 else {}
            output_dir = analysis_results[1] if len(analysis_results) > 1 else "data/output"
        else:
            stats = analysis_results or {}
            output_dir = stats.get("output_dir", "data/output")

        os.makedirs(output_dir, exist_ok=True)
        durations = stats.get("durations_s", [])

        # Histogram of rally durations.
        fig = plt.figure(figsize=(8, 5))
        if durations:
            bins = min(max(len(durations), 5), 20)
            plt.hist(durations, bins=bins, color="#3A86FF", edgecolor="black", alpha=0.85)
        else:
            # Empty-case figure still saved so pipeline output is predictable.
            plt.text(0.5, 0.5, "No rally durations available", ha="center", va="center")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)

        plt.xlabel("Rally Duration (seconds)")
        plt.ylabel("Count")
        plt.title("Distribution of Rally Durations")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        histogram_path = os.path.join(output_dir, "rally_duration_histogram.png")
        plt.savefig(histogram_path, dpi=160)
        plt.close(fig)

        return {
            "rally_duration_histogram": histogram_path,
        }