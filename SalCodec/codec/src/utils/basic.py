def geti(lamb):
    if lamb == 2048:
        return 'H265L20'
    elif lamb == 1024:
        return 'H265L23'
    elif lamb == 512:
        return 'H265L26'
    elif lamb == 256:
        return 'H265L29'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)