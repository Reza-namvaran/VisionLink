"""Map recognized gestures to Arduino serial commands — FR-8."""

from visionlink.models import Face


def gestures_to_commands(faces: list[Face]) -> list[str]:
    """Convert detected faces to a list of high-level serial commands."""
    if not faces:
        return ["NO_FACE"]

    commands = ["FACE_FOUND"]
    gestures = faces[0].gestures

    if gestures.get("smile"):
        commands.append("SMILE")
    if gestures.get("mouth_open"):
        commands.append("MOUTH_OPEN")
    if gestures.get("both_eyes_closed"):
        commands.append("EYES_CLOSED")
    elif gestures.get("left_eye") == "closed":
        commands.append("LEFT_EYE_CLOSED")
    elif gestures.get("right_eye") == "closed":
        commands.append("RIGHT_EYE_CLOSED")

    head_pose = gestures.get("head_pose", "center")
    if head_pose == "left":
        commands.append("LEFT")
    elif head_pose == "right":
        commands.append("RIGHT")
    elif head_pose == "up":
        commands.append("UP")
    elif head_pose == "down":
        commands.append("DOWN")

    return commands
