from __future__ import annotations

import pandas as pd

from pyportflowopt.output_writing import write_output_csv


def test_write_output_csv_writes_without_index(tmp_path):
    df = pd.DataFrame({"perfWndwNum": [1, 2], "SimpYrExpR": [0.1, 0.2]})
    path = tmp_path / "out.csv"
    write_output_csv(path, df)

    content = path.read_text(encoding="utf-8")
    assert content.splitlines()[0] == "perfWndwNum,SimpYrExpR"
    round_tripped = pd.read_csv(path)
    pd.testing.assert_frame_equal(round_tripped, df)
