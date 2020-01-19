#  put file 'subchallenge_1_template_data.csv' in current folder path
# function list2df:
# input:
#      5 lists:erk, akt, s6, her2, plcg2
# output:
#      df


import pandas as pd
import operator


def list2df(erk, akt, s6, her2, plcg2):
    col = ["glob_cellID", "cell_line", "treatment", "time", "cellID", "fileID", "p.ERK", "p.Akt.Ser473.", "p.S6",
           "p.HER2", "p.PLCg2"]
    tp = pd.read_csv("./subchallenge_1_template_data.csv")
    df = pd.DataFrame(columns=col)
    sub_col = ["glob_cellID", "cell_line", "treatment", "time", "cellID", "fileID"]
    df[sub_col] = tp[sub_col]
    # print(len(df))
    df.head()
    # print(len(erk))
    df["p.ERK"] = erk
    df["p.Akt.Ser473."] = akt
    df["p.S6"] = s6
    df["p.HER2"] = her2
    df["p.PLCg2"] = plcg2
    df.head()
    return df


# erk = [1]*2383058
# akt = [1]*2383058
# s6 = [1]*2383058
# her2 = [1]*2383058
# plcg2 = [1]*2383058
# df = list2df(erk, akt, s6, her2, plcg2)
#df.to_csv("./output/subchallenge_1_data_reindex.csv")
def test_format():
    col = ["glob_cellID", "cell_line", "treatment", "time", "cellID", "fileID", "p.ERK", "p.Akt.Ser473.", "p.S6",
            "p.HER2", "p.PLCg2"]
    tp = pd.read_csv("./subchallenge_1_template_data.csv")
    rs = pd.read_csv('./submit/subchallenge_1_data.csv')
    # rs = rs[col]
    # rs.to_csv("./submit/subchallenge_1_data.csv", index=False)
    print(len(tp),len(rs))
    print('#'*50)
    if not operator.eq(len(tp), len(rs)):
        print("ERROR! Different length.")
    if not all(operator.eq(tp.columns, rs.columns)):
        print("ERROR! Different columns.")
    # print("template head:", tp.head())
    # print("rs head':", rs.head())
    # print("template tail:", tp.tail())
    # print("rs tail:", rs.tail())
    if not all(operator.eq(tp.head(), rs.head())):
        print("ERROR! unmatched head")
    if not all(operator.eq(tp.tail(), rs.tail())):
        print("ERROR! unmatched tail")
    if not all(tp['glob_cellID'] == rs['glob_cellID']):
        print(rs[(tp['glob_cellID'] == rs['glob_cellID']) == False].head())
        print(tp[(tp['glob_cellID'] == rs['glob_cellID']) == False].head())
        print("ERROR! unmatched glob_cellID")
    if not all(tp['cell_line'] == rs['cell_line']):
        print("ERROR! unmatched cell_line")

    if not all(tp['treatment'] == rs['treatment']):
        print("ERROR! unmatched treatment")
    if not all(tp['time'] == rs['time']):
        print("ERROR! unmatched time")
    if not all(tp['cellID'] == rs['cellID']):
        print("ERROR! unmatched cellID")
    if not all(tp['fileID'] == rs['fileID']):
        print("ERROR! unmatched fileID")


# test_format()