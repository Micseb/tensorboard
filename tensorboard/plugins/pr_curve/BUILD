# Description:
# TensorBoard plugin for precision-recall curves.

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# The public Python api.
py_library(
  name = "api",
  srcs = ["__init__.py"],
  srcs_version = "PY2AND3",
  deps = [
      ":constants",
      "//tensorboard:expect_numpy_installed",
      "//tensorboard:expect_tensorflow_installed"
  ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorboard:internal",
    ],
)

py_library(
    name = "pr_curve_plugin",
    srcs = ["pr_curve_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorboard:internal",
    ],
    deps = [
        "//tensorboard/backend:http_util",
        "//tensorboard/backend/event_processing:event_accumulator",
        "//tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_test(
    name = "pr_curve_plugin_test",
    size = "small",
    srcs = ["pr_curve_plugin_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":api",
        ":pr_curve_plugin",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/backend:application",
        "//tensorboard/backend/event_processing:event_multiplexer",
        "//tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)