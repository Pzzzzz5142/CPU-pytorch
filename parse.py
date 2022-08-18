from bs4 import BeautifulSoup


with open("a.a", "r") as fl:
    sp = BeautifulSoup(fl, "lxml")

    cat = {"news": [], "con": [], "bio": [], "eco": [], "social": []}

    for i in sp.find_all("doc"):
        id_ = i.attrs["id"]
        if "CLIENT" in id_:
            for line in i.find_all("seg"):
                cat["con"].append(line.text.strip())
        elif "social" in id_:
            for line in i.find_all("seg"):
                cat["social"].append(line.text.strip())
        elif (
            "xinhua" in id_
            or "international_times" in id_
            or "voa-voachine" in id_
            or "hunan" in id_
            or "jingj" in id_
            or "nhan" in id_
            or "macao" in id_
            or "dw" in id_
        ):
            for line in i.find_all("seg"):
                cat["news"].append(line.text.strip())
        elif "ecommerce" in id_:
            for line in i.find_all("seg"):
                cat["eco"].append(line.text.strip())
        elif "doc" in id_ or "parallel" in id_:
            for line in i.find_all("seg"):
                cat["bio"].append(line.text.strip())
        else:
            raise Exception(i.attrs["id"])

    with open("all", "w") as fl2:
        res = ""
        for i, j in cat.items():
            with open(i, "w") as fl1:
                fl1.write("\n".join(j))
                fl1.write("\n")
            res += "\n".join(j)
            res += "\n"
        fl2.write(res)
