import time
from multiprocessing import Pool
import requests
from tqdm import tqdm
from joblib import Parallel, delayed

def get_result(question: str) -> None:
    url = "http://127.0.0.1:8080/nlpinfer"
    web = requests.post(url, json={
        "sentence": question
    })
    res = web.json()
    return res 

def main1():
    print(get_result("hello"))


def main2():

    total_N = 1000
    s1 = time.time()

    _ = [get_result(f"hello{i}") for i in tqdm(range(total_N))]
    s2 = time.time()

    es = s2 - s1
    print(f"total_N : {total_N}, use total time: {es}, qps: {total_N / es :.3f}")



def main3():
    total_N = 1000
    s1 = time.time()

    # _ = [get_result(f"hello{i}") for i in tqdm(range(total_N))]
    _ = Parallel(n_jobs=30)(delayed(get_result)(f"hello world welcome to my computer{i}") for i in tqdm(range(total_N)))
    s2 = time.time()

    es = s2 - s1
    print(f"total_N : {total_N}, use total time: {es}, qps: {total_N / es :.3f}")


if __name__ == "__main__":
    main2()
    

