import jax.tree_util as jtu


def parse(string):
    tokens = string.split("___")
    dictionary = dict([tuple(token.split(":")) for token in tokens])
    dictionary = jtu.tree_map(lambda x: x, dictionary)  # sort dictionary
    return dictionary


def unparse(dictionary):
    # print("before", dictionary)
    dictionary = jtu.tree_map(lambda x: x, dictionary)  # sort dictionary
    # print("after", dictionary)
    tokens = [f"{key}:{value}" for key, value in dictionary.items()]
    # print("tokens", tokens)
    string = "___".join(tokens)
    # print("string", string)
    return string
