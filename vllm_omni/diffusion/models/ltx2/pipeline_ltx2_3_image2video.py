# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 image-to-video pipeline aliases.

See :mod:`pipeline_ltx2_3` for details on the alias approach.
"""

from .pipeline_ltx2 import (
    get_ltx2_post_process_func,  # noqa: F401 - loaded by registry via getattr
)
from .pipeline_ltx2_image2video import (
    LTX2ImageToVideoPipeline,
    LTX2ImageToVideoTwoStagesPipeline,
)


class LTX23ImageToVideoPipeline(LTX2ImageToVideoPipeline):
    """LTX-2.3 image-to-video pipeline.

    Identical to :class:`LTX2ImageToVideoPipeline`.
    """

    pass


class LTX23ImageToVideoTwoStagesPipeline(LTX2ImageToVideoTwoStagesPipeline):
    """LTX-2.3 two-stage image-to-video pipeline.

    Identical to :class:`LTX2ImageToVideoTwoStagesPipeline`.
    """

    pass
