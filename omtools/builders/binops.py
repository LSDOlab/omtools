import numbers

from lsdo_utils.api import LinearCombinationComp, PowerCombinationComp

"""
Builders for creating binary expressions from binary
operations.
"""

# TODO: group expressions by parentheses
# TODO: create builders for expressions where input shapes do not match
# output shapes (e.g. generalized einsum)
# TODO: take ad vantage of lsdo_utils and generalize to any number of args


class Plus():
    """
    Create a LinearCombinationComp for addition.
    """
    def __init__(self, expr1, expr2):
        e1 = expr1.name
        e2 = expr2.name
        if len(expr1.builders) > 1:
            e1 = "BO_" + expr1.name + "_BC"
        if len(expr2.builders) > 1:
            e2 = "BO_" + expr2.name + "_BC"

        self.name = e1 + "_PLUS_" + e2
        if expr1.shape == expr2.shape:
            self.shape = expr1.shape

            def build(name):
                return LinearCombinationComp(
                    shape=expr1.shape,
                    out_name=name,
                    in_names=[expr1.name, expr2.name],
                    coeffs=1,
                )

            self.build = build
        else:
            print(expr1.shape)
            print(expr2.shape)
            raise ValueError("Shapes do not match")



class Minus():
    """
    Create a LinearCombinationComp for subtraction
    """
    def __init__(self, expr1, expr2):
        e1 = expr1.name
        e2 = expr2.name
        if len(expr1.builders) > 1:
            e1 = "BO_" + expr1.name + "_BC"
        if len(expr2.builders) > 1:
            e2 = "BO_" + expr2.name + "_BC"

        self.name = e1 + "_MINUS_" + e2
        if expr1.shape == expr2.shape:
            self.shape = expr1.shape

            def build(name):
                return LinearCombinationComp(
                    shape=expr1.shape,
                    out_name=name,
                    in_names=[expr1.name, expr2.name],
                    coeffs=1,
                )

            self.build = build


class Times():
    """
    Create a PowerCombinationComp for multiplication
    """
    def __init__(self, expr1, expr2):
        e1 = expr1.name
        e2 = expr2.name
        if len(expr1.builders) > 1:
            e1 = "BO_" + expr1.name + "_BC"
        if len(expr2.builders) > 1:
            e2 = "BO_" + expr2.name + "_BC"

        self.name = e1 + "_TIMES_" + e2
        if expr1.shape == expr2.shape:
            self.shape = expr1.shape

            def build(name):
                return PowerCombinationComp(
                    shape=expr1.shape,
                    out_name=name,
                    in_names=[expr1.name, expr2.name],
                )

            self.build = build


class Pow():
    """
    Create a PowerCombinationComp for scalar exponent
    """
    def __init__(self, expr1, expr2):
        if isinstance(expr2, numbers.Number):
            e = expr1.name
            if len(expr1.builders) > 1:
                e = "BO_" + expr1.name + "_BC"

            self.name = e + "_POW_" + str(expr2)
            self.shape = expr1.shape

            def build(name):
                return PowerCombinationComp(
                    shape=expr1.shape,
                    out_name=name,
                    in_names=[expr1.name],
                    powers=expr2,
                )

            self.build = build
        else:
            return NotImplemented
            # e = expr1.name
            # if len(expr1.builders) > 1:
            #     e = "BO_" + expr1.name + "_BC"

            # self.name = e + "_POW_" + str(expr2)
            # self.shape = expr1.shape

            # def build(name):
            #     return PowerCombinationComp(
            #         shape=expr1.shape,
            #         out_name=name,
            #         in_names=[expr1.name],
            #         powers=expr2,
            #     )

            # self.build = build
