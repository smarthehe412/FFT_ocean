#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <sstream>
#include <fstream>
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>

#include <GL/glew.h>

#include "glm-0.9.2.6/glm/glm.hpp"
#include "glm-0.9.2.6/glm/gtc/matrix_transform.hpp"
#include "glm-0.9.2.6/glm/gtc/type_ptr.hpp"

#include "src/keyboard.h"
#include "src/joystick.h"
#include "src/mouse.h"
#include "src/glhelper.h"
#include "src/timer.h"
#include "src/misc.h"
#include "src/obj.h"
#include "src/complex.h"
#include "src/vector.h"
#include "src/buffer.h"
#include "src/fft.h"
#include "src/cuFFT.h"

// ==================== 新增: LOD系统 ====================
const int MAX_LOD_LEVELS = 6;
const float LOD_DISTANCES[MAX_LOD_LEVELS] = {50.0f, 100.0f, 200.0f, 400.0f, 800.0f, 1600.0f};
const float LOD_TRANSITION_RANGE = 0.2f; // 20%过渡范围

// LOD统计
struct LODStats {
    int totalNodes;
    int renderedNodes;
    int lodCount[MAX_LOD_LEVELS];
};

// 四叉树节点
class QuadTreeNode {
	glm::vec3 lastCameraPos; // 跟踪上一帧相机位置
public:
    QuadTreeNode(glm::vec3 center, float size, int level, QuadTreeNode* parent = nullptr)
        : center(center), size(size), level(level), parent(parent),
          lastCameraPos(0.0f, 0.0f, 0.0f)  // 初始化 lastCameraPos
    {
        for (int i = 0; i < 4; i++) children[i] = nullptr;
        active = false;
        inFrustum = false;
    }

    ~QuadTreeNode() {
        for (int i = 0; i < 4; i++) {
            if (children[i]) delete children[i];
        }
    }

    void update(const glm::vec3& cameraPos, float frustum[6][4], int depth=0) {
		const int MAX_RECURSION_DEPTH = 20;

		if (depth > MAX_RECURSION_DEPTH) {
			active = inFrustum;
			return;
		}

		// 定义 LOD 阈值
		float lodThreshold = LOD_DISTANCES[level] * (1.0f - LOD_TRANSITION_RANGE);
		float nextLodThreshold = (level < MAX_LOD_LEVELS - 1) ? 
								LOD_DISTANCES[level + 1] : 
								FLT_MAX;

		// 保留子节点状态
		bool hadChildren = (children[0] != nullptr);
		
		// 计算距离和视锥体可见性
		float distance = glm::distance(cameraPos, center);
		
		inFrustum = isInFrustum(frustum);
		
		// 动态LOD阈值（考虑移动速度）
		float speed = glm::length(cameraPos - lastCameraPos);
		float dynamicThreshold = lodThreshold * glm::clamp(1.0f + speed * 0.1f, 1.0f, 1.5f);
		
		// 判断是否细分
		bool shouldSubdivide = inFrustum && 
							(level < MAX_LOD_LEVELS - 1) && 
							(distance < dynamicThreshold) && 
							(size > 16.0f);
		
		// 处理子节点
		if (shouldSubdivide) {
			if (!children[0]) subdivide();
			for (int i = 0; i < 4; i++) {
				children[i]->update(cameraPos, frustum, depth+1);
			}
			active = false;
		} else {
			// 智能保留策略
			bool keepChildren = false;
			if (hadChildren) {
				// 如果相机正朝该节点移动，保留子节点
				glm::vec3 toNode = glm::normalize(center - cameraPos);
				glm::vec3 moveDir = glm::normalize(cameraPos - lastCameraPos);
				
				if (glm::dot(moveDir, toNode) > 0.3f && distance < LOD_DISTANCES[level] * 2.0f) {
					keepChildren = true;
					for (int i = 0; i < 4; i++) {
						if (children[i]) {
							children[i]->update(cameraPos, frustum, depth+1);
						}
					}
				}
			}
			
			if (!keepChildren) {
				for (int i = 0; i < 4; i++) {
					if (children[i]) {
						delete children[i];
						children[i] = nullptr;
					}
				}
				active = inFrustum && (distance < nextLodThreshold * 1.2f); // 20%安全边界
			}
		}
		
		// 保存当前相机位置
		lastCameraPos = cameraPos;
	}

    void collectNodes(std::vector<QuadTreeNode*>& nodes, LODStats& stats) {
        if (!inFrustum) return;

        if (active) {
            nodes.push_back(this);
            stats.renderedNodes++;
            stats.lodCount[level]++;
        } else {
            for (int i = 0; i < 4; i++) {
                if (children[i]) children[i]->collectNodes(nodes, stats);
            }
        }
        stats.totalNodes++;
    }

    void getCorners(glm::vec3 corners[4]) const {
		float halfSize = size * 0.5f;
		
		corners[0] = center + glm::vec3(halfSize, 0.0, halfSize);
		corners[1] = center + glm::vec3(-halfSize, 0.0, halfSize);
		corners[2] = center + glm::vec3(halfSize, 0.0 , -halfSize);
		corners[3] = center + glm::vec3(-halfSize, 0.0 , -halfSize);
    }

private:
    void subdivide() {
		if (level >= MAX_LOD_LEVELS - 1) return;
        float quarterSize = size * 0.25f;
        children[0] = new QuadTreeNode(center + glm::vec3(-quarterSize, 0.0f, -quarterSize), 
                                      size * 0.5f, level + 1, this);
        children[1] = new QuadTreeNode(center + glm::vec3(quarterSize, 0.0f, -quarterSize), 
                                      size * 0.5f, level + 1, this);
        children[2] = new QuadTreeNode(center + glm::vec3(quarterSize, 0.0f, quarterSize), 
                                      size * 0.5f, level + 1, this);
        children[3] = new QuadTreeNode(center + glm::vec3(-quarterSize, 0.0f, quarterSize), 
                                      size * 0.5f, level + 1, this);
    }

    bool isInFrustum(float frustum[6][4]) const {
		glm::vec3 corners[4];
		getCorners(corners);

		for (int plane = 0; plane < 6; plane++) {

			bool inside = false;
			for (int i = 0; i < 4; i++) {
				float distance = frustum[plane][0] * corners[i].x +
								 frustum[plane][1] * corners[i].y +
								 frustum[plane][2] * corners[i].z +
								 frustum[plane][3];
				if (distance >= 0) {
					inside = true;
					break;
				}
			}
			if (!inside) return false;
		}
		return true;
    }

public:
    glm::vec3 center;
    float size;
    int level;
    QuadTreeNode* parent;
    QuadTreeNode* children[4];
    bool active;
    bool inFrustum;
};

// 提取视锥体平面
void extractFrustumPlanes(const glm::mat4& mvp, float frustum[6][4]) {
    const float* m = glm::value_ptr(mvp);
    
    // 右平面
    frustum[0][0] = m[3] - m[0];
    frustum[0][1] = m[7] - m[4];
    frustum[0][2] = m[11] - m[8];
    frustum[0][3] = m[15] - m[12];
    
    // 左平面
    frustum[1][0] = m[3] + m[0];
    frustum[1][1] = m[7] + m[4];
    frustum[1][2] = m[11] + m[8];
    frustum[1][3] = m[15] + m[12];
    
    // 下平面
    frustum[2][0] = m[3] + m[1];
    frustum[2][1] = m[7] + m[5];
    frustum[2][2] = m[11] + m[9];
    frustum[2][3] = m[15] + m[13];
    
    // 上平面
    frustum[3][0] = m[3] - m[1];
    frustum[3][1] = m[7] - m[5];
    frustum[3][2] = m[11] - m[9];
    frustum[3][3] = m[15] - m[13];
    
    // 远平面
    frustum[4][0] = m[3] - m[2];
    frustum[4][1] = m[7] - m[6];
    frustum[4][2] = m[11] - m[10];
    frustum[4][3] = m[15] - m[14];
    
    // 近平面
    frustum[5][0] = m[3] + m[2];
    frustum[5][1] = m[7] + m[6];
    frustum[5][2] = m[11] + m[10];
    frustum[5][3] = m[15] + m[14];

    // 归一化平面
    for (int i = 0; i < 6; i++) {
        float length = sqrtf(frustum[i][0]*frustum[i][0] + 
                       		 frustum[i][1]*frustum[i][1] + 
                       		 frustum[i][2]*frustum[i][2]);
        if (length > 0.0f) {
            frustum[i][0] /= length;
            frustum[i][1] /= length;
            frustum[i][2] /= length;
            frustum[i][3] /= length;
        }
    }
}

// ==================== 海洋渲染类 ====================
struct vertex_ocean {
    GLfloat   x,   y,   z; // vertex
    GLfloat  nx,  ny,  nz; // normal
    GLfloat   a,   b,   c; // htilde0
    GLfloat  _a,  _b,  _c; // htilde0mk conjugate
    GLfloat  ox,  oy,  oz; // original position
    GLfloat alpha_weight;
};

struct complex_vector_normal {   // structure used with discrete fourier transform
    complex h;      // wave height
    vector2 D;      // displacement
    vector3 n;      // normal
};

class cOcean {
private:
    bool geometry;               // flag to render geometry or surface

    float g;                // gravity constant
    float transparency;
    int N, Nplus1;              // dimension -- N should be a power of 2
    float A;                // phillips spectrum parameter -- affects heights of waves
    vector2 w;              // wind parameter
    float length;               // length parameter
    complex *h_tilde,           // for fast fourier transform
        *h_tilde_slopex, *h_tilde_slopez,
        *h_tilde_dx, *h_tilde_dz;
    cFFT *fft;              // fast fourier transform
    cuFFT *cufft;

    vertex_ocean *vertices;         // vertices for vertex buffer object
    unsigned int *indices;          // indicies for vertex buffer object
    unsigned int indices_count;     // number of indices to render
    GLuint vbo_vertices, vbo_indices;   // vertex buffer objects

    GLuint glProgram, glShaderV, glShaderF; // shaders
    GLint vertex, normal, texture, light_position, projection, view, model;  // attributes and uniforms
    GLint alpha, u_transparency; // Alpha

    GLuint seabedVAO, seabedVBO;

    GLuint foamTexture;
    GLuint normalMapTexture;
    GLint u_foamTexture;
    GLint u_normalMap;
    GLint u_time;
    
    // 新增: LOD相关
    QuadTreeNode* lodRoot;
    LODStats lodStats;
    bool debugLOD;
    GLuint lodDebugProgram;
    
    // 新增: 海平面
    float seaLevel;

protected:
public:
    cOcean(const int N, const float A, const vector2 w, const float length, bool geometry);
    ~cOcean();
    void release();
    
    void setTransparency(float t);
    void initSeabed();
    float dispersion(int n_prime, int m_prime);      // deep water
    float phillips(int n_prime, int m_prime);        // phillips spectrum
    complex hTilde_0(int n_prime, int m_prime);
    complex hTilde(float t, int n_prime, int m_prime);
    complex_vector_normal h_D_and_n(vector2 x, float t);
    void evaluateWaves(float t);
    void evaluateWavesFFT(float t);
    void render(float t, glm::vec3 light_pos, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, bool use_fft);

    GLuint generateTexture();
    
    void initTextures() {
        // 生成泡沫纹理
        glGenTextures(1, &foamTexture);
        generateTexture(); // 首次调用生成泡沫纹理
        
        // 生成法线贴图
        glGenTextures(1, &normalMapTexture);
        generateTexture(); // 再次调用生成法线贴图
        
        // 获取uniform位置
        u_foamTexture = glGetUniformLocation(glProgram, "u_foamTexture");
        u_normalMap = glGetUniformLocation(glProgram, "u_normalMap");
        u_time = glGetUniformLocation(glProgram, "u_time");
        
        // 新增: 初始化LOD根节点
        lodRoot = new QuadTreeNode(glm::vec3(0.0f, 0.0f, 0.0f), length, 0);
        debugLOD = false;
        
        // 新增: 创建LOD调试着色器
        createProgram(lodDebugProgram, glShaderV, glShaderF, "src/lod_debugv.sh", "src/lod_debugf.sh");
    }
    
    // 新增: 更新LOD系统
    void updateLOD(const glm::vec3& cameraPos, const glm::mat4& mvp) {
        // 重置统计
        memset(&lodStats, 0, sizeof(LODStats));
        
        // 提取视锥体平面
        float frustum[6][4];
        extractFrustumPlanes(mvp, frustum);
        
        // 更新LOD树
        lodRoot->update(cameraPos, frustum);
    }
    
    // 新增: 切换LOD调试显示
    void toggleDebugLOD() { debugLOD = !debugLOD; }
    
    // 新增: 渲染LOD调试信息
    void renderLODDebug(glm::mat4 Projection, glm::mat4 View, glm::mat4 Model) {
        if (!debugLOD) return;
        
        std::vector<QuadTreeNode*> nodes;
        lodRoot->collectNodes(nodes, lodStats);
        
        glUseProgram(lodDebugProgram);
        GLint projectionLoc = glGetUniformLocation(lodDebugProgram, "Projection");
        GLint viewLoc = glGetUniformLocation(lodDebugProgram, "View");
        GLint modelLoc = glGetUniformLocation(lodDebugProgram, "Model");
        
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(Projection));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(View));
        
        // 设置不同LOD级别的颜色
        glm::vec4 lodColors[MAX_LOD_LEVELS] = {
            glm::vec4(1.0f, 0.0f, 0.0f, 0.5f), // L0: 红
            glm::vec4(0.0f, 1.0f, 0.0f, 0.5f), // L1: 绿
            glm::vec4(0.0f, 0.0f, 1.0f, 0.5f), // L2: 蓝
            glm::vec4(1.0f, 1.0f, 0.0f, 0.5f), // L3: 黄
            glm::vec4(1.0f, 0.0f, 1.0f, 0.5f), // L4: 紫
            glm::vec4(0.0f, 1.0f, 1.0f, 0.5f)  // L5: 青
        };
        
        GLint colorLoc = glGetUniformLocation(lodDebugProgram, "color");
        
        // 渲染每个活动节点
        for (QuadTreeNode* node : nodes) {
            glm::vec3 corners[4];
            node->getCorners(corners);
            
            // 设置模型矩阵
            glm::mat4 nodeModel = glm::translate(Model, glm::vec3(0.0f, 0.1f, 0.0f));
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(nodeModel));
            
            // 设置颜色
            glUniform4fv(colorLoc, 1, glm::value_ptr(lodColors[node->level]));
            
            // 绘制边界框
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < 4; i++) {
                glVertex3f(corners[i].x, corners[i].y, corners[i].z);
            }
            // glVertex3f(corners[0].x, corners[0].y, corners[0].z);
            glEnd();
            
            // 绘制对角线
            glBegin(GL_LINES);
            glVertex3f(corners[0].x, corners[0].y, corners[0].z);
            glVertex3f(corners[2].x, corners[2].y, corners[2].z);
            glVertex3f(corners[1].x, corners[1].y, corners[1].z);
            glVertex3f(corners[3].x, corners[3].y, corners[3].z);
            glEnd();
        }
    }
    
    // 新增: 获取LOD统计
    const LODStats& getLODStats() const { return lodStats; }

};

GLuint cOcean::generateTexture() {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    // 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // 创建简单的程序化纹理
    unsigned char data[64*64*4]; // 64x64 RGBA纹理
    
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            int idx = (y * 64 + x) * 4;
            
            // 泡沫纹理 - 白色噪点
            if (textureID == foamTexture) {
                float noise = static_cast<float>(rand()) / RAND_MAX;
                data[idx] = 255;     // R
                data[idx+1] = 255;   // G
                data[idx+2] = 255;   // B
                data[idx+3] = static_cast<unsigned char>(noise * 200); // A (透明度)
            }
            // 法线贴图 - 蓝色噪点
            else {
                float nx = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
                float ny = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
                float nz = 1.0f; // 主要向上
                
                // 归一化
                float len = sqrt(nx*nx + ny*ny + nz*nz);
                nx /= len; ny /= len; nz /= len;
                
                data[idx] = static_cast<unsigned char>((nx + 1.0f) * 127.5f);   // R
                data[idx+1] = static_cast<unsigned char>((ny + 1.0f) * 127.5f); // G
                data[idx+2] = static_cast<unsigned char>((nz + 1.0f) * 127.5f); // B
                data[idx+3] = 255; // A
            }
        }
    }
    
    // 上传纹理数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    return textureID;
}

void cOcean::initSeabed() {
    glm::vec3 seabedColor = glm::vec3(0.2f, 0.5f, 0.3f);
    float seabedDepth = -10.0f;

    float seabedSize = length * 2.0f;
    float vertices[] = {
        -seabedSize, seabedDepth, -seabedSize,
            seabedSize, seabedDepth, -seabedSize,
            seabedSize, seabedDepth,  seabedSize,
        -seabedSize, seabedDepth,  seabedSize
    };

    glGenVertexArrays(1, &seabedVAO);
    glGenBuffers(1, &seabedVBO);
    glBindVertexArray(seabedVAO);
    glBindBuffer(GL_ARRAY_BUFFER, seabedVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

cOcean::cOcean(const int N, const float A, const vector2 w, const float length, const bool geometry) :
    g(9.81), geometry(geometry), N(N), Nplus1(N+1), A(A), w(w), length(length), transparency(0.5f),
    vertices(0), indices(0), h_tilde(0), h_tilde_slopex(0), h_tilde_slopez(0), h_tilde_dx(0), h_tilde_dz(0), fft(0), cufft(0),
    lodRoot(nullptr), seaLevel(0.0f)
{
    h_tilde        = new complex[N*N];
    h_tilde_slopex = new complex[N*N];
    h_tilde_slopez = new complex[N*N];
    h_tilde_dx     = new complex[N*N];
    h_tilde_dz     = new complex[N*N];
    vertices       = new vertex_ocean[Nplus1*Nplus1];
    indices        = new unsigned int[Nplus1*Nplus1*10];

    // if (N >= 256) cufft = new cuFFT(N);
    // fft = new cFFT(N);
    cufft = new cuFFT(N);

    int index;

    complex htilde0, htilde0mk_conj;
    for (int m_prime = 0; m_prime < Nplus1; m_prime++) {
        for (int n_prime = 0; n_prime < Nplus1; n_prime++) {
            index = m_prime * Nplus1 + n_prime;

            htilde0        = hTilde_0( n_prime,  m_prime);
            htilde0mk_conj = hTilde_0(-n_prime, -m_prime).conj();

            vertices[index].a  = htilde0.a;
            vertices[index].b  = htilde0.b;
            vertices[index]._a = htilde0mk_conj.a;
            vertices[index]._b = htilde0mk_conj.b;

            vertices[index].ox = vertices[index].x =  (n_prime - N / 2.0f) * length / N;
            vertices[index].oy = vertices[index].y =  0.0f;
            vertices[index].oz = vertices[index].z =  (m_prime - N / 2.0f) * length / N;

            vertices[index].nx = 0.0f;
            vertices[index].ny = 1.0f;
            vertices[index].nz = 0.0f;
        }
    }

    indices_count = 0;
    for (int m_prime = 0; m_prime < N; m_prime++) {
        for (int n_prime = 0; n_prime < N; n_prime++) {
            index = m_prime * Nplus1 + n_prime;

            if (geometry) {
                indices[indices_count++] = index;                // lines
                indices[indices_count++] = index + 1;
                indices[indices_count++] = index;
                indices[indices_count++] = index + Nplus1;
                indices[indices_count++] = index;
                indices[indices_count++] = index + Nplus1 + 1;
                if (n_prime == N - 1) {
                    indices[indices_count++] = index + 1;
                    indices[indices_count++] = index + Nplus1 + 1;
                }
                if (m_prime == N - 1) {
                    indices[indices_count++] = index + Nplus1;
                    indices[indices_count++] = index + Nplus1 + 1;
                }
            } else {
                indices[indices_count++] = index;                // two triangles
                indices[indices_count++] = index + Nplus1;
                indices[indices_count++] = index + Nplus1 + 1;
                indices[indices_count++] = index;
                indices[indices_count++] = index + Nplus1 + 1;
                indices[indices_count++] = index + 1;
            }
        }
    }

    createProgram(glProgram, glShaderV, glShaderF, "src/oceanv.sh", "src/oceanf.sh");

    vertex         = glGetAttribLocation(glProgram, "vertex");
    normal         = glGetAttribLocation(glProgram, "normal");
    texture        = glGetAttribLocation(glProgram, "texture");
    light_position = glGetUniformLocation(glProgram, "light_position");
    projection     = glGetUniformLocation(glProgram, "Projection");
    view           = glGetUniformLocation(glProgram, "View");
    model          = glGetUniformLocation(glProgram, "Model");

    alpha          = glGetAttribLocation(glProgram, "aAlphaWeight");
    u_transparency = glGetUniformLocation(glProgram, "uTransparency");

    glGenBuffers(1, &vbo_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_ocean)*(Nplus1)*(Nplus1), vertices, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &vbo_indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_count*sizeof(unsigned int), indices, GL_STATIC_DRAW);
    
    // 初始化LOD系统
    initTextures();
	// std::cerr << "Debug: Reached line " << __LINE__  << std::endl;
}

cOcean::~cOcean() {
    if (h_tilde)        delete [] h_tilde;
    if (h_tilde_slopex) delete [] h_tilde_slopex;
    if (h_tilde_slopez) delete [] h_tilde_slopez;
    if (h_tilde_dx)     delete [] h_tilde_dx;
    if (h_tilde_dz)     delete [] h_tilde_dz;
    if (fft)        delete fft;
    if (cufft)      delete cufft;
    if (vertices)       delete [] vertices;
    if (indices)        delete [] indices;
    if (lodRoot)        delete lodRoot;
}

void cOcean::release() {
    glDeleteVertexArrays(1, &seabedVAO);
    glDeleteBuffers(1, &seabedVBO);
    glDeleteBuffers(1, &vbo_indices);
    glDeleteBuffers(1, &vbo_vertices);
    releaseProgram(glProgram, glShaderV, glShaderF);
    if (lodDebugProgram) glDeleteProgram(lodDebugProgram);
}

void cOcean::setTransparency(float t) {
    transparency = glm::clamp(0.0f, 1.0f, t);
}

float cOcean::dispersion(int n_prime, int m_prime) {
    float w_0 = 2.0f * M_PI / 200.0f;
    float kx = M_PI * (2 * n_prime - N) / length;
    float kz = M_PI * (2 * m_prime - N) / length;
    return floor(sqrt(g * sqrt(kx * kx + kz * kz)) / w_0) * w_0;
}

float cOcean::phillips(int n_prime, int m_prime) {
    vector2 k(M_PI * (2 * n_prime - N) / length,
          M_PI * (2 * m_prime - N) / length);
    float k_length  = k.length();
    if (k_length < 0.000001) return 0.0;

    float k_length2 = k_length  * k_length;
    float k_length4 = k_length2 * k_length2;

    float k_dot_w   = k.unit() * w.unit();
    float k_dot_w2  = k_dot_w * k_dot_w * k_dot_w * k_dot_w * k_dot_w * k_dot_w;

    float w_length  = w.length();
    float L         = w_length * w_length / g;
    float L2        = L * L;
    
    float damping   = 0.001;
    float l2        = L2 * damping * damping;

        float directionalFocus = 2.0f * pow(abs(k_dot_w), 4.0f);
    float steepness = 0.8f - 0.3f * exp(-k_length2 * L2 * 4.0f);

    return A * exp(-1.0f/(k_length2 * L2)) / k_length4 
             * k_dot_w2 * directionalFocus * steepness
             * exp(-k_length2 * l2);
}

complex cOcean::hTilde_0(int n_prime, int m_prime) {
    complex r = gaussianRandomVariable();
    return r * sqrt(phillips(n_prime, m_prime) / 2.0f);
}

complex cOcean::hTilde(float t, int n_prime, int m_prime) {
    int index = m_prime * Nplus1 + n_prime;

    complex htilde0(vertices[index].a,  vertices[index].b);
    complex htilde0mkconj(vertices[index]._a, vertices[index]._b);

    float omegat = dispersion(n_prime, m_prime) * t;

    float cos_ = cos(omegat);
    float sin_ = sin(omegat);

    complex c0(cos_,  sin_);
    complex c1(cos_, -sin_);

    complex res = htilde0 * c0 + htilde0mkconj * c1;

    return htilde0 * c0 + htilde0mkconj*c1;
}

complex_vector_normal cOcean::h_D_and_n(vector2 x, float t) {
    complex h(0.0f, 0.0f);
    vector2 D(0.0f, 0.0f);
    vector3 n(0.0f, 0.0f, 0.0f);

    complex c, res, htilde_c;
    vector2 k;
    float kx, kz, k_length, k_dot_x;

    for (int m_prime = 0; m_prime < N; m_prime++) {
        kz = 2.0f * M_PI * (m_prime - N / 2.0f) / length;
        for (int n_prime = 0; n_prime < N; n_prime++) {
            kx = 2.0f * M_PI * (n_prime - N / 2.0f) / length;
            k = vector2(kx, kz);

            k_length = k.length();
            k_dot_x = k * x;

            c = complex(cos(k_dot_x), sin(k_dot_x));
            htilde_c = hTilde(t, n_prime, m_prime) * c;

            h = h + htilde_c;

            n = n + vector3(-kx * htilde_c.b, 0.0f, -kz * htilde_c.b);

            if (k_length < 0.000001) continue;
            D = D + vector2(kx / k_length * htilde_c.b, kz / k_length * htilde_c.b);
        }
    }
    
    n = (vector3(0.0f, 1.0f, 0.0f) - n).unit();

    complex_vector_normal cvn;
    cvn.h = h;
    cvn.D = D;
    cvn.n = n;
    return cvn;
}

void cOcean::evaluateWaves(float t) {
    float lambda = -1.0;
    int index;
    vector2 x;
    vector2 d;
    complex_vector_normal h_d_and_n;
    for (int m_prime = 0; m_prime < N; m_prime++) {
        for (int n_prime = 0; n_prime < N; n_prime++) {
            index = m_prime * Nplus1 + n_prime;

            x = vector2(vertices[index].x, vertices[index].z);

            h_d_and_n = h_D_and_n(x, t);

            vertices[index].y = h_d_and_n.h.a;

            vertices[index].x = vertices[index].ox + lambda*h_d_and_n.D.x;
            vertices[index].z = vertices[index].oz + lambda*h_d_and_n.D.y;

            vertices[index].nx = h_d_and_n.n.x;
            vertices[index].ny = h_d_and_n.n.y;
            vertices[index].nz = h_d_and_n.n.z;

            if (n_prime == 0 && m_prime == 0) {
                vertices[index + N + Nplus1 * N].y = h_d_and_n.h.a;
            
                vertices[index + N + Nplus1 * N].x = vertices[index + N + Nplus1 * N].ox + lambda*h_d_and_n.D.x;
                vertices[index + N + Nplus1 * N].z = vertices[index + N + Nplus1 * N].oz + lambda*h_d_and_n.D.y;

                vertices[index + N + Nplus1 * N].nx = h_d_and_n.n.x;
                vertices[index + N + Nplus1 * N].ny = h_d_and_n.n.y;
                vertices[index + N + Nplus1 * N].nz = h_d_and_n.n.z;
            }
            if (n_prime == 0) {
                vertices[index + N].y = h_d_and_n.h.a;
            
                vertices[index + N].x = vertices[index + N].ox + lambda*h_d_and_n.D.x;
                vertices[index + N].z = vertices[index + N].oz + lambda*h_d_and_n.D.y;

                vertices[index + N].nx = h_d_and_n.n.x;
                vertices[index + N].ny = h_d_and_n.n.y;
                vertices[index + N].nz = h_d_and_n.n.z;
            }
            if (m_prime == 0) {
                vertices[index + Nplus1 * N].y = h_d_and_n.h.a;
            
                vertices[index + Nplus1 * N].x = vertices[index + Nplus1 * N].ox + lambda*h_d_and_n.D.x;
                vertices[index + Nplus1 * N].z = vertices[index + Nplus1 * N].oz + lambda*h_d_and_n.D.y;
                
                vertices[index + Nplus1 * N].nx = h_d_and_n.n.x;
                vertices[index + Nplus1 * N].ny = h_d_and_n.n.y;
                vertices[index + Nplus1 * N].nz = h_d_and_n.n.z;
            }
        }
    }
}

void cOcean::evaluateWavesFFT(float t) {
    float kx, kz, len, lambda = -1.0f;
    int index, index1;

    for (int m_prime = 0; m_prime < N; m_prime++) {
        kz = M_PI * (2.0f * m_prime - N) / length;
        for (int n_prime = 0; n_prime < N; n_prime++) {
            kx = M_PI*(2 * n_prime - N) / length;
            len = sqrt(kx * kx + kz * kz);
            index = m_prime * N + n_prime;

            h_tilde[index] = hTilde(t, n_prime, m_prime);
            h_tilde_slopex[index] = h_tilde[index] * complex(0, kx);
            h_tilde_slopez[index] = h_tilde[index] * complex(0, kz);
            if (len < 0.000001f) {
                h_tilde_dx[index]     = complex(0.0f, 0.0f);
                h_tilde_dz[index]     = complex(0.0f, 0.0f);
            } else {
                h_tilde_dx[index]     = h_tilde[index] * complex(0, -kx/len);
                h_tilde_dz[index]     = h_tilde[index] * complex(0, -kz/len);
            }
        }
    }

    if (cufft) {
        cufft->batch_fft(h_tilde, true);
        cufft->batch_fft(h_tilde_slopex, true);
        cufft->batch_fft(h_tilde_slopez, true);
        cufft->batch_fft(h_tilde_dx, true);
        cufft->batch_fft(h_tilde_dz, true);

        cufft->batch_fft(h_tilde, false);
        cufft->batch_fft(h_tilde_slopex, false);
        cufft->batch_fft(h_tilde_slopez, false);
        cufft->batch_fft(h_tilde_dx, false);
        cufft->batch_fft(h_tilde_dz, false);
    }
    else {
        for (int m_prime = 0; m_prime < N; m_prime++) {
            fft->fft(h_tilde, h_tilde, 1, m_prime * N);
            fft->fft(h_tilde_slopex, h_tilde_slopex, 1, m_prime * N);
            fft->fft(h_tilde_slopez, h_tilde_slopez, 1, m_prime * N);
            fft->fft(h_tilde_dx, h_tilde_dx, 1, m_prime * N);
            fft->fft(h_tilde_dz, h_tilde_dz, 1, m_prime * N);
        }
        for (int n_prime = 0; n_prime < N; n_prime++) {
            fft->fft(h_tilde, h_tilde, N, n_prime);
            fft->fft(h_tilde_slopex, h_tilde_slopex, N, n_prime);
            fft->fft(h_tilde_slopez, h_tilde_slopez, N, n_prime);
            fft->fft(h_tilde_dx, h_tilde_dx, N, n_prime);
            fft->fft(h_tilde_dz, h_tilde_dz, N, n_prime);
        }
    }

    int sign;
    float signs[] = { 1.0f, -1.0f };
    vector3 n;
    for (int m_prime = 0; m_prime < N; m_prime++) {
        for (int n_prime = 0; n_prime < N; n_prime++) {
            index  = m_prime * N + n_prime;     // index into h_tilde..
            index1 = m_prime * Nplus1 + n_prime;    // index into vertices

            sign = signs[(n_prime + m_prime) & 1];

            h_tilde[index]     = h_tilde[index] * sign;

            // height
            vertices[index1].y = h_tilde[index].a;

            // displacement
            h_tilde_dx[index] = h_tilde_dx[index] * sign;
            h_tilde_dz[index] = h_tilde_dz[index] * sign;
            vertices[index1].x = vertices[index1].ox + h_tilde_dx[index].a * lambda;
            vertices[index1].z = vertices[index1].oz + h_tilde_dz[index].a * lambda;
            
            // normal
            h_tilde_slopex[index] = h_tilde_slopex[index] * sign;
            h_tilde_slopez[index] = h_tilde_slopez[index] * sign;
            n = vector3(0.0f - h_tilde_slopex[index].a, 1.0f, 0.0f - h_tilde_slopez[index].a).unit();
            vertices[index1].nx =  n.x;
            vertices[index1].ny =  n.y;
            vertices[index1].nz =  n.z;
            vertices[index1].alpha_weight = glm::smoothstep(-1.0f, 1.0f, -vertices[index1].y);

            // for tiling
            if (n_prime == 0 && m_prime == 0) {
                vertices[index1 + N + Nplus1 * N].y = h_tilde[index].a;

                vertices[index1 + N + Nplus1 * N].x = vertices[index1 + N + Nplus1 * N].ox + h_tilde_dx[index].a * lambda;
                vertices[index1 + N + Nplus1 * N].z = vertices[index1 + N + Nplus1 * N].oz + h_tilde_dz[index].a * lambda;
            
                vertices[index1 + N + Nplus1 * N].nx =  n.x;
                vertices[index1 + N + Nplus1 * N].ny =  n.y;
                vertices[index1 + N + Nplus1 * N].nz =  n.z;
                vertices[index1 + N + Nplus1 * N].alpha_weight = glm::smoothstep(-1.0f, 1.0f, -vertices[index1 + N + Nplus1 * N].y);
            }
            if (n_prime == 0) {
                vertices[index1 + N].y = h_tilde[index].a;

                vertices[index1 + N].x = vertices[index1 + N].ox + h_tilde_dx[index].a * lambda;
                vertices[index1 + N].z = vertices[index1 + N].oz + h_tilde_dz[index].a * lambda;
            
                vertices[index1 + N].nx =  n.x;
                vertices[index1 + N].ny =  n.y;
                vertices[index1 + N].nz =  n.z;
                vertices[index1 + N].alpha_weight = glm::smoothstep(-1.0f, 1.0f, vertices[index1 + N].y);
            }
            if (m_prime == 0) {
                vertices[index1 + Nplus1 * N].y = h_tilde[index].a;

                vertices[index1 + Nplus1 * N].x = vertices[index1 + Nplus1 * N].ox + h_tilde_dx[index].a * lambda;
                vertices[index1 + Nplus1 * N].z = vertices[index1 + Nplus1 * N].oz + h_tilde_dz[index].a * lambda;
            
                vertices[index1 + Nplus1 * N].nx =  n.x;
                vertices[index1 + Nplus1 * N].ny =  n.y;
                vertices[index1 + Nplus1 * N].nz =  n.z;
                vertices[index1 + Nplus1 * N].alpha_weight = glm::smoothstep(-1.0f, 1.0f, -vertices[index1 + Nplus1 * N].y);
            }
        }
    }
}

void renderSkyGradient() {
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glBegin(GL_QUADS);
    // 顶部 - 深天蓝
    glColor3f(0.4f, 0.7f, 0.9f);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f(1.0f, -1.0f);
    
    // 底部 - 浅天蓝
    glColor3f(0.53f, 0.81f, 0.92f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
    
    glEnable(GL_DEPTH_TEST);
}

void cOcean::render(float t, glm::vec3 light_pos, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, bool use_fft) {
    static bool eval = false;
    if (!use_fft && !eval) {
        eval = true;
        evaluateWaves(t);
    } else if (use_fft) {
        evaluateWavesFFT(t);
    }
    
    // 更新LOD系统
    glm::mat4 mvp = Projection * View * Model;
    glm::mat4 invView = glm::inverse(View);
    glm::vec3 cameraPos = glm::vec3(invView[3]);
    updateLOD(cameraPos, mvp);

    glClearColor(0.53f, 0.81f, 0.92f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderSkyGradient();

    glUseProgram(glProgram);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); 

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, foamTexture);
    glUniform1i(u_foamTexture, 1);
    
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, normalMapTexture);
    glUniform1i(u_normalMap, 2);
    
    // 设置时间
    glUniform1f(u_time, t);

    float lightDistance = glm::distance(light_pos, cameraPos);
    GLint u_lightDistance_loc = glGetUniformLocation(glProgram, "u_lightDistance");
    if (u_lightDistance_loc != -1) {
        glUniform1f(u_lightDistance_loc, lightDistance);
    } else {
        // fprintf(stderr, "u_lightDistance uniform not found\n");
    }

    glUniform1f(u_transparency, transparency);
    glUniform3f(light_position, light_pos.x, light_pos.y, light_pos.z);
    glUniformMatrix4fv(projection, 1, GL_FALSE, glm::value_ptr(Projection));
    glUniformMatrix4fv(view,       1, GL_FALSE, glm::value_ptr(View));
    glUniformMatrix4fv(model,      1, GL_FALSE, glm::value_ptr(Model));

    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertex_ocean) * Nplus1 * Nplus1, vertices);
    glEnableVertexAttribArray(vertex);
    glVertexAttribPointer(vertex, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean), 0);
    glEnableVertexAttribArray(normal);
    glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean), (char *)NULL + 12);
    glEnableVertexAttribArray(texture);
    glVertexAttribPointer(texture, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean), (char *)NULL + 24);
    glEnableVertexAttribArray(alpha);
    glVertexAttribPointer(alpha, 1, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean),
                        (void*)offsetof(vertex_ocean, alpha_weight));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
    
    // 收集要渲染的LOD节点
    std::vector<QuadTreeNode*> nodes;
    lodRoot->collectNodes(nodes, lodStats);
    
    // 渲染所有活动节点
    for (QuadTreeNode* node : nodes) {
        // 计算节点位置和缩放
        float scaleFactor = 4.0f / powf(2.0f, node->level);
        glm::mat4 nodeModel = glm::translate(Model, node->center);
        nodeModel = glm::scale(nodeModel, glm::vec3(scaleFactor));
        
        // 设置模型矩阵
        glUniformMatrix4fv(model, 1, GL_FALSE, glm::value_ptr(nodeModel));
        
        // 根据LOD级别调整细节
        int lodDetail = N / (1 << node->level);
        if (lodDetail < 4) lodDetail = 4;
        
        // 绘制节点
        glDrawElements(geometry ? GL_LINES : GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, 0);
    }


    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    
    // 渲染LOD调试信息
    renderLODDebug(Projection, View, Model);
}

void renderLODStats(const LODStats& stats, int x, int y, SDL_Surface* screen, TTF_Font* font) {
    // 颜色定义
    SDL_Color white = {255, 255, 255, 0};
    SDL_Color black = {0, 0, 0, 0};
    
    // 创建背景矩形（SDL1.2方式）
    SDL_Rect backgroundRect = {x, y, 300, 150};
    SDL_FillRect(screen, &backgroundRect, SDL_MapRGB(screen->format, black.r, black.g, black.b));
    
    // 渲染标题
    SDL_Surface* titleSurface = TTF_RenderText_Solid(font, "LOD Statistics", white);
    SDL_Rect titleRect = {x + 10, y + 110, 0, 0};
    SDL_BlitSurface(titleSurface, NULL, screen, &titleRect);
    SDL_FreeSurface(titleSurface);
    
    // 渲染统计数据
    std::stringstream ss;
    ss << "Total Nodes: " << stats.totalNodes << "\n"
       << "Rendered Nodes: " << stats.renderedNodes << "\n";
    
    for (int i = 0; i < MAX_LOD_LEVELS; i++) {
        ss << "LOD " << i << ": " << stats.lodCount[i] << "\n";
    }
    
    int line = 1;
    std::string lineStr;
    while (std::getline(ss, lineStr)) {
        SDL_Surface* textSurface = TTF_RenderText_Solid(font, lineStr.c_str(), white);
        SDL_Rect textRect = {x + 10, y + 130 + line * 20, 0, 0};
        SDL_BlitSurface(textSurface, NULL, screen, &textRect);
        SDL_FreeSurface(textSurface);
        line++;
    }
}

class OceanTile {
public:
    OceanTile(const glm::vec3& center, float size, int N, float A, const vector2& w)
        : center(center), size(size), baseN(N), A(A), w(w), 
          ocean(N, A, w, size, false), currentLOD(0) {
        modelMatrix = glm::translate(glm::mat4(1.0f), center);
    }
    
    void update(const glm::vec3& cameraPos, float frustum[6][4], 
               std::vector<OceanTile*>& visibleTiles) {
        // 使用SDL获取时间
        float currentTime = SDL_GetTicks() / 1000.0f;
        ocean.evaluateWavesFFT(currentTime);

		updateLOD(cameraPos);
        
        // 更新LOD
        updateLOD(cameraPos);
        
        // 检查可见性
        visible = isVisible(frustum);
        if (visible) visibleTiles.push_back(this);
    }
    
    void render(float t, const glm::vec3& lightPos, 
                const glm::mat4& VP, const glm::mat4& view) {
        if (!visible) return;
        
        ocean.render(t, lightPos, VP, view, modelMatrix, true);
    }
    
    const glm::vec3& getCenter() const { return center; }
    float getSize() const { return size; }
    
    void updateLOD(const glm::vec3& cameraPos) {
        float distance = glm::distance(cameraPos, center);
        int newLOD = 0;
        
        // 根据距离确定LOD级别
        if (distance > 2000.0f) newLOD = 3;
        else if (distance > 1000.0f) newLOD = 2;
        else if (distance > 500.0f) newLOD = 1;
        
        // 如果LOD变化，重新创建海洋
        if (newLOD != currentLOD) {
            int newN = baseN / (1 << newLOD);
            if (newN < 16) newN = 16; // 最小网格尺寸
            
            ocean = cOcean(newN, A, w, size, false);
            currentLOD = newLOD;
        }
    }
    
    bool isVisible(float frustum[6][4]) const {
        // 检查四个角点
        glm::vec3 corners[4] = {
            center + glm::vec3(-size/2, 0, -size/2),
            center + glm::vec3(size/2, 0, -size/2),
            center + glm::vec3(size/2, 0, size/2),
            center + glm::vec3(-size/2, 0, size/2)
        };
        
        for (int plane = 0; plane < 6; plane++) {
            bool inside = false;
            for (int i = 0; i < 4; i++) {
                float distance = 
                    frustum[plane][0] * corners[i].x +
                    frustum[plane][1] * corners[i].y +
                    frustum[plane][2] * corners[i].z +
                    frustum[plane][3];
                if (distance >= 0) {
                    inside = true;
                    break;
                }
            }
            if (!inside) return false;
        }
        return true;
    }

    glm::vec3 center;
    float size;
    int baseN;
    float A;
    vector2 w;
    int currentLOD;
    cOcean ocean;
    glm::mat4 modelMatrix;
    bool visible = true;
};

class InfiniteOcean {
public:
    InfiniteOcean(int baseN, float baseSize, float A, const vector2& w)
        : baseN(baseN), baseSize(baseSize), A(A), w(w) {
        // 初始创建中心瓦片
        allTiles.push_back(new OceanTile(glm::vec3(0, 0, 0), baseSize, baseN, A, w));
    }

    void update(const glm::vec3& cameraPos, const glm::mat4& VP) {
        // 提取视锥体
        float frustum[6][4];
        extractFrustumPlanes(VP, frustum);
        
        // 更新可见瓦片
        visibleTiles.clear();
        for (auto tile : allTiles) {
            tile->update(cameraPos, frustum, visibleTiles);
			
        	glm::vec3 center = tile->center;
        	float distance = glm::distance(cameraPos, center);
			std::cerr << "Tile at (" << center.x << ", " << center.z 
                  << ") size: " << tile->getSize() 
                  << ", distance: " << distance << "\n";
        }
        
        // 添加新瓦片
        addNewTiles(cameraPos, frustum);
    }

    void render(float t, const glm::vec3& lightPos, 
                const glm::mat4& VP, const glm::mat4& view) {
        for (auto& tile : visibleTiles) {
            tile->render(t, lightPos, VP, view);
        }
    }
    
    void toggleDebugLOD() { debugLOD = !debugLOD; }
    
    const LODStats& getLODStats() const {
        // 简单统计
        lodStats.totalNodes = allTiles.size();
        lodStats.renderedNodes = visibleTiles.size();
        return lodStats;
    }
    
	~InfiniteOcean() {
        release();
    }
    
    void release() {
        std::cout << "Releasing " << allTiles.size() << " tiles..." << std::endl;
        for (auto tile : allTiles) {
            if (tile) {
                delete tile;
                tile = nullptr;
            }
        }
        allTiles.clear();
        visibleTiles.clear();
    }
    
    // 禁用拷贝和赋值
    InfiniteOcean(const InfiniteOcean&) = delete;
    InfiniteOcean& operator=(const InfiniteOcean&) = delete;

private:
    void addNewTiles(const glm::vec3& cameraPos, float frustum[6][4]) {
        // 计算相机所在的瓦片坐标
        int tileX = static_cast<int>(cameraPos.x / baseSize);
        int tileZ = static_cast<int>(cameraPos.z / baseSize);
        
        // 检查周围3x3区域的瓦片
        const int radius = 1;
        for (int z = tileZ - radius; z <= tileZ + radius; z++) {
            for (int x = tileX - radius; x <= tileX + radius; x++) {
                glm::vec3 center(x * baseSize, 0, z * baseSize);
                
                // 如果瓦片不存在且可见，创建新瓦片
                if (!tileExists(center) && isInFrustum(center, frustum)) {
                    allTiles.push_back(new OceanTile(center, baseSize, baseN, A, w));
                }
            }
        }
    }
    
    bool tileExists(const glm::vec3& center) const {
        for (auto tile : allTiles) {
            if (tile->getCenter() == center) return true;
        }
        return false;
    }
    
    bool isInFrustum(const glm::vec3& center, float frustum[6][4]) const {
        // 简化的可见性检查
        for (int i = 0; i < 6; i++) {
            float distance = frustum[i][0] * center.x + 
                             frustum[i][1] * center.y + 
                             frustum[i][2] * center.z + 
                             frustum[i][3];
            if (distance < -baseSize) return false;
        }
        return true;
    }

    std::vector<OceanTile*> allTiles;
    std::vector<OceanTile*> visibleTiles;
    int baseN;
    float baseSize;
    float A;
    vector2 w;
    bool debugLOD = false;
    mutable LODStats lodStats; // 可变的统计对象
};


int main(int argc, char *argv[]) {

    TTF_Init();
	TTF_Font* font = TTF_OpenFont("ttf/arial.ttf", 16);

    // constants
    const int WIDTH  = 1600, HEIGHT = 900;
    
    // buffer for grabs
    cBuffer buffer(WIDTH, HEIGHT);

    // controls
    cKeyboard kb; int key_up, key_down, key_front, key_back, key_left, key_right, keyx, keyy, keyz;
    cJoystick js; joystick_position jp[2];
    cMouse    ms; mouse_state mst;

    // timers
    cTimer t0; double elapsed0; cTimer t1; double elapsed1; cTimer t2; double elapsed2; cTimer video; double elapsed_video;

    // application is active.. fullscreen flag.. screen grab.. video grab..
    bool active = true, fullscreen = false, screen_grab = false, video_grab = false;

    // setup an opengl context.. initialize extension wrangler
    SDL_Surface *screen = my_SDL_init(WIDTH, HEIGHT, fullscreen);
    SDL_Event event;

    // ocean simulator
    // cOcean ocean(128, 0.0005f, vector2(32.0f,32.0f), 64, false);
    // ocean.setTransparency(1.0);
    // ocean.initTextures();

	InfiniteOcean ocean(128, 1024.0f, 0.0005f, vector2(32.0f, 32.0f));

    // model view projection matrices and light position
    glm::mat4 Projection = glm::perspective(45.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f); 
    glm::mat4 View       = glm::mat4(1.0f);
    glm::mat4 Model      = glm::mat4(1.0f);
    glm::vec3 light_position;

    // rotation angles and viewpoint
    float alpha =   0.0f, beta =   0.0f, gamma =   0.0f,
          pitch =   0.0f, yaw  =   0.0f, roll  =   0.0f,
          x     =   0.0f, y    =   0.0f, z     = -20.0f;

    key_up = key_down = key_front = key_back = key_left = key_right = 0;
    int step_length = 100;
    int use_fft = 1;
    float zoom_scale = 1.0;
    while(active) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                active = false;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                case SDLK_g: screen_grab  = true; break;
                case SDLK_v: video_grab  ^= true; elapsed_video = 0.0; break;
                case SDLK_f:
                    fullscreen ^= true;
                    screen = SDL_SetVideoMode(WIDTH, HEIGHT, 32, (fullscreen ? SDL_FULLSCREEN : 0) | SDL_HWSURFACE | SDL_OPENGL);
                    break;
                case SDLK_e: use_fft ^= 1; break;
                case SDLK_m: zoom_scale = 1.0; break;
                case SDLK_l: ocean.toggleDebugLOD(); break; // 切换LOD调试显示
                case SDLK_SPACE: key_up = 1; break;
                case SDLK_LSHIFT: key_down = 1; break;
                case SDLK_w: key_front = 1; break;
                case SDLK_s: key_back = 1; break;
                case SDLK_a: key_left = 1; break;
                case SDLK_d: key_right = 1; break;
                }
                break;
            case SDL_KEYUP:
                switch (event.key.keysym.sym) {
                case SDLK_SPACE: key_up = 0; break;
                case SDLK_LSHIFT: key_down = 0; break;
                case SDLK_w: key_front = 0; break;
                case SDLK_s: key_back = 0; break;
                case SDLK_a: key_left = 0; break;
                case SDLK_d: key_right = 0; break;
                }
                break;
            case SDL_MOUSEMOTION:
                mst.axis[0] = event.motion.x;
                mst.axis[1] = event.motion.y;
                break;
            case SDL_MOUSEBUTTONDOWN:
                switch(event.button.button) {
                    case SDL_BUTTON_WHEELUP: zoom_scale *= 1.1; break;
                    case SDL_BUTTON_WHEELDOWN: zoom_scale *= 0.9; break;
                    case SDL_BUTTON_LEFT: zoom_scale *= 1.5; break;
                    case SDL_BUTTON_RIGHT: zoom_scale *= 0.5; break;
                }
                if (zoom_scale <= 0.5) zoom_scale = 0.5;
                if (zoom_scale >= 10.0) zoom_scale = 10.0;
                // std::cout << zoom_scale <<std::endl;
                break;
            }
        }


        // time elapsed since last frame
        elapsed0 = t0.elapsed(true);

        // update frame based on input state
        /*if (kb.getKeyState(KEY_UP))    alpha += 180.0f*elapsed0;
        if (kb.getKeyState(KEY_DOWN))  alpha -= 180.0f*elapsed0;
        if (kb.getKeyState(KEY_LEFT))  beta  -= 180.0f*elapsed0;
        if (kb.getKeyState(KEY_RIGHT)) beta  += 180.0f*elapsed0;
        jp[0] = js.joystickPosition(0);
        jp[1] = js.joystickPosition(1);
        yaw   += jp[1].x*elapsed0*90;
        pitch += jp[1].y*elapsed0*90;
        x     += -cos(-yaw*M_PI/180.0f)*jp[0].x*elapsed0*30 + sin(-yaw*M_PI/180.0f)*jp[0].y*elapsed0*30;
        z     +=  cos(-yaw*M_PI/180.0f)*jp[0].y*elapsed0*30 + sin(-yaw*M_PI/180.0f)*jp[0].x*elapsed0*30;
        */
        // mst = ms.getMouseState();
        yaw   =  (mst.axis[0] - WIDTH / 2) * 0.2;
        pitch = -(mst.axis[1] - HEIGHT / 2) * 0.2;

        // key_up    = kb.getKeyState(KEY_W);
        // key_down  = kb.getKeyState(KEY_S);
        // key_left  = kb.getKeyState(KEY_A);
        // key_right = kb.getKeyState(KEY_D);
        keyx = -key_left +  key_right;
        keyz =  key_front   + -key_back;
        keyy = -key_up + key_down;
        x     += -cos(-yaw*M_PI/180.0f)*keyx*elapsed0*step_length + sin(-yaw*M_PI/180.0f)*keyz*elapsed0*step_length;
        z     +=  cos(-yaw*M_PI/180.0f)*keyz*elapsed0*step_length + sin(-yaw*M_PI/180.0f)*keyx*elapsed0*step_length;
        y     +=  keyy*elapsed0*step_length;

        // rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // viewing and light position
        View  = glm::mat4(1.0f);
        View  = glm::rotate(View, pitch, glm::vec3(-1.0f, 0.0f, 0.0f));
        View  = glm::rotate(View, yaw,   glm::vec3(0.0f, 1.0f, 0.0f));
        View  = glm::translate(View, glm::vec3(x, -50 + y, z));
        light_position = glm::vec3(1000.0f, 100.0f, -1000.0f);

        Projection = glm::perspective(45.0f / zoom_scale, (float)WIDTH / (float)HEIGHT, 0.1f * zoom_scale, 1000.0f * zoom_scale); 

        // ocean.render(t1.elapsed(false), light_position, Projection, View, Model, use_fft);
		glm::mat4 VP = Projection * View;
		glm::mat4 mvp = Projection * View * Model;
		glm::mat4 invView = glm::inverse(View);
		glm::vec3 cameraPos = glm::vec3(invView[3]);
		ocean.update(cameraPos, VP);
		ocean.render(SDL_GetTicks() / 1000.0f, light_position, VP, View);
        std::cerr << "Debug: Reached line " << __LINE__  << std::endl;

        // 显示LOD统计信息
        renderLODStats(ocean.getLODStats(), 10, 10, screen, font);
		std::cerr << "Debug: Reached line " << __LINE__  << std::endl;

        SDL_GL_SwapBuffers();

        if (screen_grab) { screen_grab = false; buffer.save(false); }
    }

    ocean.release();

	TTF_CloseFont(font);
	TTF_Quit();
    SDL_Quit();

    return 0;
}