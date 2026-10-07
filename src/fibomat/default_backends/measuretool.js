// source-sha256: d9c403c0f987d0099880cedb40fde81e0b5e29b0fc1ae9f959c8dd30389adf53
"use strict";
var _a;
Object.defineProperty(exports, "__esModule", { value: true });
exports.MeasureTool = exports.MeasureToolView = void 0;
// Measure tool for bokeh plots: drag with the mouse to measure the distance and the angle between two points.
//
// This file is compiled by bokeh (see `scripts/build_measuretool.py`, which creates `measuretool.js`). Never edit
// `measuretool.js` by hand.
const gesture_tool_1 = require("models/tools/gestures/gesture_tool");
const column_data_source_1 = require("models/sources/column_data_source");
const label_1 = require("models/annotations/label");
class MeasureToolView extends gesture_tool_1.GestureToolView {
    constructor() {
        super(...arguments);
        this.start = null;
    }
    _to_data(sx, sy) {
        const { frame } = this.plot_view;
        if (!frame.bbox.contains(sx, sy))
            return null;
        return [frame.x_scale.invert(sx), frame.y_scale.invert(sy)];
    }
    _reset() {
        this.start = null;
        this.model.source.data = { x: [], y: [] };
        this.model.label.text = "";
    }
    _pan_start(ev) {
        this.start = this._to_data(ev.sx, ev.sy);
    }
    _pan(ev) {
        if (this.start == null)
            return;
        const end = this._to_data(ev.sx, ev.sy);
        if (end == null)
            return;
        const [x0, y0] = this.start;
        const [x1, y1] = end;
        const dx = x1 - x0;
        const dy = y1 - y0;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angle_rad = Math.atan2(dy, dx);
        const angle_deg = angle_rad * 180 / Math.PI;
        const unit = this.model.measure_unit == "" ? "" : " " + this.model.measure_unit;
        this.model.source.data = { x: [x0, x1], y: [y0, y1] };
        this.model.label.text =
            "distance = " + distance.toFixed(3) + unit +
                ", angle = " + angle_deg.toFixed(2) + "° (" + angle_rad.toFixed(2) + " rad)";
    }
    _pan_end(_ev) {
        this.start = null;
    }
    // a click removes the measurement
    _tap(_ev) {
        this._reset();
    }
    deactivate() {
        super.deactivate();
        this._reset();
    }
}
exports.MeasureToolView = MeasureToolView;
MeasureToolView.__name__ = "MeasureToolView";
class MeasureTool extends gesture_tool_1.GestureTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Measure";
        this.tool_icon = "bk-tool-icon-crosshair";
        this.event_type = ["pan", "tap"];
        this.default_order = 12;
    }
}
exports.MeasureTool = MeasureTool;
_a = MeasureTool;
MeasureTool.__name__ = "MeasureTool";
(() => {
    _a.prototype.default_view = MeasureToolView;
    _a.define(({ Ref, Str }) => ({
        source: [Ref(column_data_source_1.ColumnDataSource)],
        label: [Ref(label_1.Label)],
        measure_unit: [Str, ""],
    }));
})();
//# sourceMappingURL=measuretool.js.map
