"""Tests of the plots in a real browser (headless chromium, driven with playwright).

They are skipped if playwright (``pip install playwright``, a browser download is not needed) or a chromium/chrome
executable is not available.
"""
import shutil

import pytest

from fibomat.default_backends import BokehBackend
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d, two_d
from fibomat.shapes import Line, Rect
from fibomat.units import unit

sync_api = pytest.importorskip('playwright.sync_api')

BROWSER = next(
    (path for path in (shutil.which(name) for name in ('chromium', 'chromium-browser', 'google-chrome')) if path), None
)
pytestmark = pytest.mark.skipif(BROWSER is None, reason='no chromium or chrome executable found')


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


@pytest.fixture(scope='module')
def plot_file(tmp_path_factory):
    layout = Layout(description='browser test')
    site = layout.create_site(um(0, 0))
    site.create_pattern(
        Line((0, 0), (2, 1)) * unit('µm'), Mill(1. * unit('ms'), 1),
        one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    )
    site.create_pattern(
        Rect(2, 1).translated((3, 3)) * unit('µm'), Mill(1. * unit('ms'), 1),
        two_d.LineByLine(
            0.25 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False,
            one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
    )
    path = tmp_path_factory.mktemp('plot') / 'plot.html'
    layout.plot(show=False, filename=path)
    return path


@pytest.fixture
def page(plot_file):
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=BROWSER, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 900, 'height': 700})
        page.messages = []
        page.requests = []
        page.on('console', lambda message: page.messages.append((message.type, message.text)))
        page.on('pageerror', lambda error: page.messages.append(('pageerror', str(error))))
        page.on('request', lambda request: page.requests.append(request.url))
        page.goto(plot_file.as_uri())
        page.wait_for_selector('canvas')
        page.wait_for_timeout(500)
        yield page
        browser.close()


def label_texts(page):
    return page.evaluate("() => [...Bokeh.documents[0].all_models].filter(m => m.type == 'Label').map(m => m.text)")


def drag(page, start, end):
    box = page.locator('canvas').first.bounding_box()
    page.mouse.move(box['x'] + start[0], box['y'] + start[1])
    page.mouse.down()
    page.mouse.move(box['x'] + end[0], box['y'] + end[1], steps=8)
    return page


def test_plot_is_rendered_without_problems(page):
    assert page.locator('canvas').count() >= 1
    problems = [message for message in page.messages if message[0] in ('error', 'warning', 'pageerror')]
    assert problems == []


def test_plot_is_self_contained(page):
    # nothing was loaded from the internet (only the file itself and data urls)
    assert [url for url in page.requests if not url.startswith(('file:', 'data:', 'blob:'))] == []


def test_measure_tool_measures_distance_and_angle(page):
    page.locator('[title="Measure"]').first.click()

    drag(page, (200, 300), (300, 200))  # 100 px to the right and 100 px up: 45 degrees (the aspect is equal)
    page.mouse.up()
    page.wait_for_timeout(200)

    texts = [text for text in label_texts(page) if text]
    assert len(texts) == 1
    assert texts[0].startswith('distance = ') and ' µm, angle = 45.00° (0.79 rad)' in texts[0]


def test_measure_tool_distance_is_in_plot_units(page):
    page.locator('[title="Measure"]').first.click()

    # the pixel size of the plot: the x axis spans an interval of known length
    scale = page.evaluate("""() => {
        const plot = Bokeh.documents[0].roots()[0];
        const view = Bokeh.index.get_one(plot);
        const frame = view.frame;
        return (frame.x_range.end - frame.x_range.start) / frame.bbox.width;
    }""")

    drag(page, (200, 300), (300, 300))
    page.mouse.up()
    page.wait_for_timeout(200)

    text = next(text for text in label_texts(page) if text)
    distance = float(text.split('distance = ')[1].split(' µm')[0])
    assert distance == pytest.approx(100 * scale, rel=0.02)
    assert 'angle = 0.00°' in text


def test_a_click_removes_the_measurement(page):
    page.locator('[title="Measure"]').first.click()
    drag(page, (200, 300), (300, 200))
    page.mouse.up()
    page.wait_for_timeout(100)
    assert any(label_texts(page))

    box = page.locator('canvas').first.bounding_box()
    page.mouse.click(box['x'] + 400, box['y'] + 400)
    page.wait_for_timeout(200)
    assert not any(label_texts(page))
