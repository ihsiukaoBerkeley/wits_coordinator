def concat_list(list1, list2):
    print(f"concating with * operator: {[*list1, *list2]}")
    cp = list1.copy()
    for i in list2:
        cp.append(i)
    print(f"concating with for-loop: {cp}")

concat_list([1,2,3,4,5],[6,7,8,9,10])
