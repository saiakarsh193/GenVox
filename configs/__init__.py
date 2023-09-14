def check_argument(name, value, min_val=None, max_val=None):
    if (min_val == None):
        assert (value <= max_val), f"The value \'{name}\' ({value}) is above max_val ({max_val})."
    elif (max_val == None):
        assert (value >= min_val), f"The value \'{name}\' ({value}) is below min_val ({min_val})."
    else:
        assert (value >= min_val and value <= max_val), f"The value \'{name}\' ({value}) is not in the required range ({min_val} -> {max_val})."


class BaseConfig:
    def __str__(self, level = 0):
        prefix_main = (("├── " + "    " * (level - 1)) if level > 0 else "")
        prefix = (("│   " + "    " * (level - 1)) if level > 0 else "")
        rstr = prefix_main + self.__class__.__name__ + "\n"
        for ind, (key, value) in enumerate(self.__dict__.items()):
            if (isinstance(value, BaseConfig)):
                svalue = "\n" + BaseConfig.__str__(value, level=level + 1)
            else:
                svalue = "(" + str(value) + ")"
            prefix_value = ("└── " if ind == len(self.__dict__) - 1 else "├── ")
            rstr += prefix + prefix_value + key.ljust(35) + svalue + "\n"
        return rstr.rstrip()
    
    def __repr__(self):
        return str(self)