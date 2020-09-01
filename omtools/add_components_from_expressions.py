def add_components_from_expressions(group, *args):
    """
    Call builder methods to create OpenMDAO components
    """
    # q = {**z(g), **y(g), **e(g)}
    # TODO: more terse way of writing this?
    builders = {}
    for expr in args:
        builders.update(expr.builders)
    for name, builder in builders.items():
        group.add_subsystem(
            name,
            builder.build(builder.name),
            promotes=['*'],
        )
