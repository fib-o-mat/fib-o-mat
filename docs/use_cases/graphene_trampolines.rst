Mechanical resonators based on suspended single-layer graphene
==============================================================

In this use case it is demonstrated, how fib-o-mat can be used to generate patterns based on a few user-set constraints.
This allows to quickly generate patterns which follow certain pre-defined rules without recreating the shape manually.
Instead, the shape is calculated according to user defined rules and constraints which may themselves depend on only a few
parameters.

In the present example, trampoline shapes should be cut out of graphene which sits on a substrate with circular holes.
The pattern geometry is depicted in the figure below. The grey shaded area is the graphene trampoline which
should remain after patterning. Everything else should be cut away. This is achieved by creating cuts on the four trampoline
arms plotted in orange. This causes the four non-trampoline areas to fold in a way that these are out of the way for
the wanted measurements.

.. figure:: /_static/trampoline.png
    :align: center
    :width: 250px

    Pattern geometry for graphene trampolines. The grey shaded area is the trampoline which should remain after
    patterning. The orange lines depict the cuttings made to achieve this. The only required parameters to generate the
    patterns are the trampoline radius and the bridge width.

The geometry per trampoline arm consists of two line segments and a circular arc (cf. the figure above). The
exact positions of these is specified only by the trampoline radius and the bridge width. By defining these the complete
pattern is generated. Additionally, the substrate hole radius must be set which was constant in this case.

The script `graphene_trampolines.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/use_cases/graphene_trampolines.py>`__ generates different variations of the trampoline pattern, changing the scale, rotation, bridge width and
trampoline radius.
