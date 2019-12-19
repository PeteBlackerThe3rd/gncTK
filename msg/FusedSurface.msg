# This message stores a reconstructed surface from a fused image and lidar scan
# A single image is included along with a texture mapped triangular mesh
#

Header header
gnc_tool_kit/Vertex[] vertices
gnc_tool_kit/Triangle[] triangles
gnc_tool_kit/TextureCoordinate[] texCoords
sensor_msgs/Image cameraImage

float64 sensor_origin_x
float64 sensor_origin_y
float64 sensor_origin_z