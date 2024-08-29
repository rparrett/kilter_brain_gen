from rich import print as pprint

# https://rich.readthedocs.io/en/stable/appendix/colors.html#appendix-colors
HOLD_META = {
    "r12": ("00FF00", "green1", "start"),
    "r13": ("00FFFF", "cyan1", "middle"),
    "r14": ("FF00FF", "magenta1", "finish"),
    "r15": ("FFA500", "orange1", "foot"),  # ffaf00 close
}
# start=12, middle=13, finish=14, foot=15


def color_markup_for_role_id(role_id, text):
    return "[bold " + HOLD_META[f"r{role_id}"][1] + "]" + text + "[/]"


def print_colored_role_names():
    for k in map(str, (12, 13, 14, 15)):
        v = HOLD_META[f"r{k}"]
        pprint(color_markup_for_role_id(k, k + " " + v[2]))
