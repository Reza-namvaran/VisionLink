"""Batch evaluation report — Phase 2."""

from collections import Counter
from pathlib import Path

from visionlink.models import AnalysisResult


def build_batch_report(results: list[AnalysisResult]) -> str:
    """Build a human-readable summary of batch processing results."""
    total = len(results)
    failed = [r for r in results if r.error]
    succeeded = [r for r in results if not r.error]
    no_faces = [r for r in succeeded if r.face_count == 0]

    gesture_counts = Counter()
    head_poses = Counter()
    total_faces = 0
    timings = [r.elapsed_ms for r in succeeded if r.elapsed_ms is not None]

    for result in succeeded:
        total_faces += result.face_count
        for face in result.faces:
            g = face.gestures
            if g.get("smile"):
                gesture_counts["smile"] += 1
            if g.get("mouth_open"):
                gesture_counts["mouth open"] += 1
            if g.get("both_eyes_closed"):
                gesture_counts["both eyes closed"] += 1
            elif g.get("left_eye") == "closed":
                gesture_counts["left eye closed"] += 1
            elif g.get("right_eye") == "closed":
                gesture_counts["right eye closed"] += 1
            head_poses[str(g.get("head_pose", "center"))] += 1

    lines = [
        "VisionLink Batch Report",
        "=" * 40,
        f"Images processed:  {total}",
        f"Succeeded:         {len(succeeded)}",
        f"Failed:            {len(failed)}",
        f"No face detected:  {len(no_faces)}",
        f"Total faces:       {total_faces}",
        "",
    ]

    if timings:
        avg_ms = sum(timings) / len(timings)
        lines.extend(
            [
                "Performance",
                "-" * 40,
                f"Average time:      {avg_ms:.0f} ms/image",
                f"Total time:        {sum(timings):.0f} ms",
                "",
            ]
        )

    if gesture_counts:
        lines.extend(["Gestures detected", "-" * 40])
        for name, count in sorted(gesture_counts.items()):
            lines.append(f"  {name:<22} {count}")
        lines.append("")

    if head_poses:
        lines.extend(["Head pose distribution", "-" * 40])
        for pose, count in sorted(head_poses.items()):
            lines.append(f"  {pose:<22} {count}")
        lines.append("")

    if failed:
        lines.extend(["Errors", "-" * 40])
        for result in failed:
            lines.append(f"  {result.source.name}: {result.error}")
        lines.append("")

    if no_faces:
        lines.extend(["Images with no faces", "-" * 40])
        for result in no_faces:
            lines.append(f"  {result.source.name}")
        lines.append("")

    return "\n".join(lines)


def save_batch_report(results: list[AnalysisResult], path: Path) -> Path:
    """Write a text report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_batch_report(results) + "\n", encoding="utf-8")
    return path


def build_batch_stats(results: list[AnalysisResult]) -> dict[str, object]:
    """Build aggregate statistics for JSON export."""
    succeeded = [r for r in results if not r.error]
    timings = [r.elapsed_ms for r in succeeded if r.elapsed_ms is not None]
    return {
        "images_processed": len(results),
        "succeeded": len(succeeded),
        "failed": sum(1 for r in results if r.error),
        "no_face_images": sum(1 for r in succeeded if r.face_count == 0),
        "total_faces": sum(r.face_count for r in succeeded),
        "avg_elapsed_ms": round(sum(timings) / len(timings), 1) if timings else None,
        "total_elapsed_ms": round(sum(timings), 1) if timings else None,
    }
