with open('index.txt', 'w') as f:
    for i in range(1, 109):
        url = f"https://ilibrary.ru/text/1544/p.{i}/index.html"
        f.write(f"{i}. {url}\n")

