#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>

namespace py = pybind11;

int pt_in_box3d(float x, float y, float z, float cx, float cy, float cz, float h, float w, float l, float cosa, float sina) {
	if ((fabsf(x - cx) > 10.) || (fabsf(y - cy) > h / 2.0) || (fabsf(z - cz) > 10.)){
			return 0;
	}

	float x_rot = (x - cx) * cosa + (z - cz) * (-sina);
	float z_rot = (x - cx) * sina + (z - cz) * cosa;

	int in_flag = static_cast<int>((x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) & (z_rot <= w / 2.0));
	return in_flag;
}

py::array_t<int> pts_in_boxes3d(py::array_t<float> pts, py::array_t<float> boxes) {
  py::buffer_info pts_buf= pts.request(), boxes_buf = boxes.request();

  if (pts_buf.ndim != 2 || boxes_buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be 2");
  }
  if (pts_buf.shape[1] != 3) {
    throw std::runtime_error("pts 2nd dimension must be 3");
  }
  if (boxes_buf.shape[1] != 7) {
    throw std::runtime_error("boxes 2nd dimension must be 7");
  }

  auto pts_num = pts_buf.shape[0];
  auto boxes_num = boxes_buf.shape[0];
  auto mask = py::array_t<int>(pts_num * boxes_num);
  py::buffer_info mask_buf = mask.request();

  float *pts_ptr = (float *) pts_buf.ptr,
        *boxes_ptr = (float *) boxes_buf.ptr;
  int *mask_ptr = (int *) mask_buf.ptr;

  for (ssize_t i = 0; i < boxes_num; i++) {
    float cx = boxes_ptr[i * 7];
    float cy = boxes_ptr[i * 7 + 1] - boxes_ptr[i * 7 + 3] / 2.;
    float cz = boxes_ptr[i * 7 + 2];
    float h = boxes_ptr[i * 7 + 3];
    float w = boxes_ptr[i * 7 + 4];
    float l = boxes_ptr[i * 7 + 5];
    float angle = boxes_ptr[i * 7 + 6];
    float cosa = cosf(angle);
    float sina = sinf(angle);
    for (ssize_t j = 0; j < pts_num; j++) {
      mask_ptr[i * pts_num + j] = pt_in_box3d(pts_ptr[j * 3], pts_ptr[j * 3 + 1], pts_ptr[j * 3 + 2], cx, cy, cz, h, w, l, cosa, sina);
    }
  }

  mask.resize({boxes_num, pts_num});
  return mask;
}

PYBIND11_MODULE(pts_utils, m) {
    m.def("pts_in_boxes3d", &pts_in_boxes3d, "Calculate mask for whether points in boxes3d");
}
