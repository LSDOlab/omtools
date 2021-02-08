from omtools.comps.rotation_matrix_comp import RotationMatrixComp
from omtools.core.expression import Expression


class rotmat(Expression):
    def initialize(self, angle, axis: str):
        if isinstance(angle, Expression):

            self.add_predecessor_node(angle)

            if angle.shape == (1, ):
                self.shape = (3, 3)

            else:
                self.shape = angle.shape + (3, 3)

            self.build = lambda: RotationMatrixComp(
                shape=angle.shape,
                in_name=angle.name,
                out_name=self.name,
                axis=axis,
            )

        else:
            raise TypeError(angle, " is not an Expression object")
