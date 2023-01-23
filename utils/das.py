from typing import List


def get_das_nodes(node_list_str: str) -> List[str]:
    # For example, node[301,303,305-320]
    das_node_list = []
    node_list_str = node_list_str[5:-1]
    parts = node_list_str.split(",")
    for part in parts:
        if "-" in part:
            subparts = part.split("-")
            das_node_list += list(range(int(subparts[0]), int(subparts[1]) + 1))
        else:
            das_node_list.append(int(part))

    return ["node%d" % node_id for node_id in das_node_list]
