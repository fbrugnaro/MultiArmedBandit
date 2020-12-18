from mab import Egreedy, ThompsonSampling, Ucb1, LinUcb
import numpy as np
import time
import fileinput
import random

def read_news_data(path):
    """    
    Parameters
    ----------
    path : path to data files
    
    Stores
    -------    
    articles : Array of unique article ids
    events : [
                 0 : displayed article index,
                 1 : reward,
                 2 : array of user features,
                 3 : array of available article indexes
             ]
    """

    global articles, features, events, arms_number, n_events
    articles = []
    features = []
    events = []

    skipped = 0

    with fileinput.input(files=path) as f:
        for line in f:
            skip = False
            cols = line.split()
            if (len(cols) - 10) % 7 != 0 and (len(cols) - 12) % 7 != 0:     #skip lines with errors
                skipped += 1
            else:
                article_indexes = []
                article_pool = []
                end_range = len(cols) - 6
                if ((len(cols) - 12) % 7 == 0):    # catch broken lines that can be fixed
                    end_range = len(cols) - 8
                id_present = False
                for i in range(10, end_range, 7):
                    id = cols[i][1:]
                    if id not in articles:
                        articles.append(id)
                        features.append([float(x[2:]) for x in cols[i + 1: i + 7]])
                    article_indexes.append(articles.index(id))
                    article_pool.append(id)
                    if id == cols[1]:       #displayed article must be in the available article pool
                        id_present = True

                if id_present:
                    events.append(
                        [
                            article_pool.index(cols[1]),
                            int(cols[2]),
                            [float(x[2:]) for x in cols[4:10]],
                            article_indexes,
                        ]
                    )
                else:
                    skipped += 1

    features = np.array(features)
    arms_number = len(articles)

    print(len(events), "events read with", arms_number, "articles")
    if skipped != 0:
        print("Skipped broken events:", skipped)


def run(bandit, train_ratio = 0.9, size=100, random=False):
    ctr_num = 0
    ctr_den = 0
    ctr_num_test = 0
    ctr_den_test = 0
    start = time.time()
    train = []
    test = []

    if size == 100:
        evs = events
    else:
        k = int(len(events) * size / 100)
        evs = random.sample(events, k)

    for t, event in enumerate(evs):
        displayed = event[0]
        reward = event[1]
        user = event[2]
        article_idx = event[3]

        if random:
            ctr_num_test += reward
            ctr_den_test += 1
            test.append(ctr_num_test/ctr_den_test)
        else:
            selected = bandit.select_arm(user, article_idx, t, features)
            if selected == displayed:
                if(np.random.rand() < train_ratio):              
                    bandit.update(displayed, reward, user, article_idx, features)
                    ctr_num += reward
                    ctr_den += 1
                    train.append(ctr_num/ctr_den)
                else:
                    ctr_num_test += reward
                    ctr_den_test += 1
                    test.append(ctr_num_test/ctr_den_test)

            

    end = time.time()

    execution_time = round(end - start, 1)
    execution_time = (
        str(round(execution_time / 60, 1)) + "m"
        if execution_time > 60
        else str(execution_time) + "s"
    )
    print(bandit.name)
    print("Execution time: ", execution_time)
    ctr = 0
    if ctr_den_test < 1:
        print("No impressions were made.")
    else:
        ctr = ctr_num_test / ctr_den_test
        print("CTR achieved: ", round(ctr, 5))
        
    return train, test

if __name__ == "__main__":
    paths = ['Data/dataset_example']
    read_news_data(paths)
    bandits = [Egreedy(0.01,arms_number)]
    for b in bandits:
        run(b)
