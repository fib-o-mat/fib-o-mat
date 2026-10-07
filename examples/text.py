from fibomat.layout import Layout
from fibomat.units import unit
from fibomat.composite_shapes import Text

hello = Text('Hello fib-o-mat!')

increased_font_size = Text('My font size is 2 units.', font_size=2)
increased_font_size = increased_font_size.translated((0, 2))

s = Layout()

s.add_annotation(hello * unit('µm'))

s.add_annotation(increased_font_size * unit('µm'))


s.plot()

