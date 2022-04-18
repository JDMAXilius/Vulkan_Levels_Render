// minimalistic code to draw a single triangle, this is not part of the API.
// TODO: Part 1b
//#include "FSLogo.h"
#include "h2bParser.h"
#include "shaderc/shaderc.h" // needed for compiling shaders at runtime
#define MAX_SUBMESH_PER_DRAW 1024
#define PLAYMUSIC false
#ifdef _WIN32 // must use MT platform DLL libraries on windows
	#pragma comment(lib, "shaderc_combined.lib") 
#endif
// include ktx texture loader, cmake will link the correct platform libs
#define KHRONOS_STATIC // must be defined if ktx libraries are built statically
#include <ktxvulkan.h>
// Simple Vertex Shader
const char* vertexShaderSource = R"(
#pragma pack_matrix(row_major)
// TODO: 2i
// an ultra simple hlsl vertex shader
// TODO: Part 2b
struct ATTRIBUTES
{
    float3 Kd; // diffuse reflectivity
    float d; // dissolve (transparency) 
    float3 Ks; // specular reflectivity
    float Ns; // specular exponent
    float3 Ka; // ambient reflectivity
    float sharpness; // local reflection map sharpness
    float3 Tf; // transmission filter
    float Ni; // optical density (index of refraction)
    float3 Ke; // emissive reflectivity
    uint illum; // illumination model
};
struct LIGHT {
	uint type;
	float radius;
	float4 pos;
	float4 color;
	float4 dir;
};
struct SHADER_MODEL_DATA
{
    float4 sunDirection, sunColor, sunAmbient, camPos;
    matrix ViewMatrix, ProjectionMatrix;
    matrix matricies[1024];
    ATTRIBUTES materials[1024];
	LIGHT light[16];
};
StructuredBuffer<SHADER_MODEL_DATA> SceneData;
// TODO: Part 4g
// TODO: Part 2i
// TODO: Part 3e
[[vk::push_constant]]
cbuffer MESH_INDEX {
	uint mesh_ID;
	uint wm_ID;
	uint Light_ID;
};
// TODO: Part 4a
// TODO: Part 1f
struct OBJ_VERT
{
    float3 pos : POSITION;
    float3 uvw : TEXCOORD0;
    float3 nrm : NORMAL;
	//float4 RGBA : COLOR;
};
struct OutputStruct
{
	float4 posH : SV_POSITION;
	float3 nrmW : NORMAL;
	float3 posW : WORLD;
	float3 uvw : TEXTCOORD0;
	//float4 RGBA : COLOR;
	//float3 tangent : TEXTCOORD0;
	//float3 bitangent : TEXTCOORD0;
};
// TODO: Part 4b
OutputStruct main(OBJ_VERT inputObj, uint instanceId : SV_INSTANCEID) : SV_POSITION
{
	OutputStruct tempStruct;
	float4 pos = float4(inputObj.pos, 1);
	tempStruct.posH = mul(pos, SceneData[0].matricies[instanceId]);
	tempStruct.nrmW = normalize(inputObj.nrm);
	tempStruct.nrmW = mul(tempStruct.nrmW, SceneData[0].matricies[instanceId]); //vec3 normal = mat3( ModelViewMatrix ) * app_normal;
	tempStruct.posW = tempStruct.posH;
	tempStruct.posH = mul(tempStruct.posH, SceneData[0].ViewMatrix);
	tempStruct.posH = mul(tempStruct.posH, SceneData[0].ProjectionMatrix); // gl_Position = ProjectionMatrix * ModelViewMatrix * app_position; 
	tempStruct.uvw = inputObj.uvw;
	
    return tempStruct;
}
)";
// Simple Pixel Shader
const char* pixelShaderSource = R"(
struct ATTRIBUTES
{
    float3 Kd; // diffuse reflectivity
    float d; // dissolve (transparency) 
    float3 Ks; // specular reflectivity
    float Ns; // specular exponent
    float3 Ka; // ambient reflectivity
    float sharpness; // local reflection map sharpness
    float3 Tf; // transmission filter
    float Ni; // optical density (index of refraction)
    float3 Ke; // emissive reflectivity
    uint illum; // illumination model
};
struct LIGHT {
	uint type;
	float radius;
	float4 pos;
	float4 color;
	float4 dir;
};
struct SHADER_MODEL_DATA
{
    float4 sunDirection, sunColor, sunAmbient, camPos;
    matrix ViewMatrix, ProjectionMatrix;
    matrix matricies[1024];
    ATTRIBUTES materials[1024];
	LIGHT light[16];
};
StructuredBuffer<SHADER_MODEL_DATA> SceneData;
// TODO: Part 4g
// TODO: Part 2i
// TODO: Part 3e
[[vk::push_constant]]
cbuffer MESH_INDEX
{
    uint mesh_ID;
    uint wm_ID;
	uint Light_ID;
};
// TODO: Part 4a
// TODO: Part 1f
struct OBJ_VERT
{
    float3 pos : POSITION;
    float3 uvw : TEXCOORD0;
    float3 nrm : NORMAL;
};
struct InputStruct
{
    float4 posH : SV_POSITION;
    float3 nrmW : NORMAL;
    float3 posW : WORLD;
    float3 uvw : TEXTCOORD0;
};

// an ultra simple hlsl pixel shader
// TODO: Part 4b

//float4 CalculateLight(vec3 lightDirection, vec3 radiance, float intensity)
//{
//    vec3 halfwayDirection = normalize(viewDirection + lightDirection);
//    float lightViewAngle = max(dot(halfwayDirection, viewDirection), 0.0f);
//    float lightAngle = max(dot(normalDirection, lightDirection), 0.0f);
//
//    float distribution = CalculateDistribution(halfwayDirection);
//    float geometry = CalculateGeometry(lightAngle);
//    vec3 fresnel = CalculateFresnel(lightViewAngle);
//
//    vec3 diffuseComponent = vec3(1.0f) - fresnel;
//    diffuseComponent *= 1.0f - metallic;
//
//    vec3 nominator = distribution * geometry * fresnel;
//    float denominator = 4 * viewAngle * lightAngle + 0.001f;
//    vec3 specularComponent = nominator / denominator;
//
//    //Return the combined components.
//    return float4((diffuseComponent * albedoColor / PI + specularComponent) * radiance * lightAngle * intensity, 1);
//}
//
//float4 CalculatePointLight(int index)
//{
//    //Calculate the point light.
//    vec3 lightDirection = normalize(pointLightWorldPositions[index] - fragmentWorldPosition);
//
//    float distanceToLightSource = length(fragmentWorldPosition - pointLightWorldPositions[index]);
//    float attenuation = clamp(1.0f - distanceToLightSource / pointLightAttenuationDistances[index], 0.0f, 1.0f);
//    attenuation *= attenuation;
//
//    vec3 radiance = pointLightColors[index] * attenuation;
//
//    return CalculateLight(lightDirection, radiance, pointLightIntensities[index]);
//}
//
//float4 CalculateSpotLight(InputStruct inputObj) : SV_TARGET
//{
//    //Calculate the spot light.
//    float3 lightDirection = normalize(SceneData[0].light.pos - inputObj.posW);
//    float angle = dot(lightDirection, -SceneData[0].light.dir);
//
//    float distanceToLightSource = length(inputObj.posW - SceneData[0].light.pos);
//    float attenuation = clamp(1.0f - distanceToLightSource / SceneData[0].light.radius, 0.0f, 1.0f);
//    attenuation *= attenuation;
//
//    float3 radiance = spotLightColors[index] * attenuation;
//
//    float3 calculatedLight = CalculateLight(lightDirection, radiance, spotLightIntensities[index]);
//
//    //Calculate the inner/outer cone fade out.
//    float epsilon = spotLightInnerCutoffAngles[index] - spotLightOuterCutoffAngles[index];
//    float intensity = angle > spotLightInnerCutoffAngles[index] ? 1.0f : clamp((angle - spotLightOuterCutoffAngles[index]) / epsilon, 0.0f, 1.0f); 
//
//    calculatedLight *= intensity;
//
//    return float4(calculatedLight,0);
//}

float4 main(InputStruct inputObj) : SV_TARGET
{
	inputObj.nrmW = normalize(inputObj.nrmW);
	float4 Normal = normalize(float4(inputObj.nrmW, 0));

	//float FANGLE = saturate(dot(Normal.xyz, -SceneData[0].sunDirection.xyz));// * SceneData[0].sunColor;
	float FANGLE = saturate(dot(Normal.xyz, -SceneData[0].sunDirection.xyz));

	float4 pointLight[2];
	float3 lightP;
	for (int i = 0; i < 2; i++) 
	{
		float3 lightDir = normalize(SceneData[0].light[i].pos - inputObj.posW);
		float attenuation = 1.0f - saturate(length(SceneData[0].light[i].pos - inputObj.posW)/SceneData[0].light[i].radius);
		float lightRatio = saturate(dot(lightDir, inputObj.nrmW)) * attenuation;
		lightP = lightRatio * SceneData[0].light[i].color;// * SceneData[0].materials[mesh_ID].Kd;
		pointLight[i] = float4(lightP, 0);
		//Light_ID = i;
	};

	float4 DirectLight = SceneData[0].sunColor * FANGLE;
	float4 Ambient = float4(0.25, 0.25, 0.35, 1.0);
	//float4 Ambient = FANGLE + SceneData[0].sunAmbient;

	float4 IndirectLight = Ambient; // * (float4(SceneData[0].materials[mesh_ID].Kd, 1.0f));

    float3 ViewDir = normalize(SceneData[0].camPos.xyz - inputObj.posW);
    float3 Halfvector = normalize((-SceneData[0].sunDirection.xyz) + ViewDir);
	float Intensity = saturate(pow(dot(Normal, Halfvector), SceneData[0].materials[mesh_ID].Ns));
    float4 ReflectedLight = float4(SceneData[0].materials[mesh_ID].Ks, 1) * Intensity;
	
	float shinniness = 60.0; 
    float specular = pow( dot( Halfvector, inputObj.nrmW ), shinniness ); 

	//float4 reflected = reflect(DirectLight, inputObj.nrmW);
	//float intensityv = pow(saturate(dot(ViewDir, reflected)), SceneData[0].materials[mesh_ID].Ns);

    float4 fLightColor = saturate( DirectLight  + IndirectLight) * (float4(SceneData[0].materials[mesh_ID].Kd, 1) + ReflectedLight + float4(SceneData[0].materials[mesh_ID].Ke , 1)) + pointLight[0];// + pointLight[1];// + specular;
	//float4 fLightColor = IndirectLight + float4(SceneData[0].sunColor.xyz * SceneData[0].materials[mesh_ID].Ks * intensityv, 0.0f);

	return fLightColor;
	
}
)";

	struct GlobalUbo {
		GW::MATH::GMATRIXF projectionView{ 1.f };
		//GW::MATH::GVECTORF lightDirection; // = glm::normalize(glm::vec3{ 1.f, -3.f, -1.f });
		GW::MATH::GVECTORF ambientLightColor{ 1.f, 1.f, 1.f, .02f };  // w is intensity
		GW::MATH::GVECTORF lightPosition{ -1.f };
		alignas(16) GW::MATH::GVECTORF lightColor{ 1.f };  // w is light intensity
	};

struct LIGHT {
	unsigned type = 16;
	float radius = 2;
	GW::MATH::GVECTORF pos = { -1,100,0,0 };
	GW::MATH::GVECTORF color = { 1.f, 1.f, 1.f, .05f };
	GW::MATH::GVECTORF dir;
};

struct SHADER_MODEL_DATA
{
	GW::MATH::GVECTORF sunDirection, sunColor, sunAmbient, camPos;//light info
	GW::MATH::GMATRIXF ViewMatrix, ProjectionMatrix;//view info
	//
	GW::MATH::GMATRIXF matricies[MAX_SUBMESH_PER_DRAW];//world sapace
	//OBJ_ATTRIBUTES materialsf[MAX_SUBMESH_PER_DRAW];
	H2B::ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];//color/texture
	//sampler2D texSampler;
	//GW::MATH::GVECTORF sunAmbient, camPos;
	LIGHT light[MAX_SUBMESH_PER_DRAW];
};

struct Model_Struct {
	std::string filename;
	std::string filepath;
	std::vector<GW::MATH::GMATRIXF> WorldMatrices;
	H2B::Parser h2bParser;
};
struct Model_INDEX {
	unsigned mesh_ID;
	unsigned wm_ID;
	unsigned Light_ID = 0;
};

struct Model_Data
{
	std::vector<H2B::VERTEX> Vertexes;
	int VertexSize;
	int IndexSize;
	std::vector<unsigned> Indexes;
	std::vector<unsigned> VertexOffsets;
	std::vector<unsigned> IndexOffsets;
	std::vector<unsigned> MatrixOffset;
	std::vector<unsigned> MaterialOffset;
};

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float uv[2];
	float normal[3];
};

class Renderer
{
	// Instance
	SHADER_MODEL_DATA ShaderModelData;
	std::map<std::string, Model_Struct> mapModels;
	Model_Data ModelData;
	std::map<std::string, Model_Struct>::iterator iter;
	Model_INDEX modelIndex;


	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GVulkanSurface vlk;
	GW::CORE::GEventReceiver shutdown;

	GW::AUDIO::GMusic GMusic;
	GW::AUDIO::GSound GSound;
	GW::AUDIO::GMusic music;
	GW::AUDIO::GMusic vxfmusic;
	GW::AUDIO::GAudio GAudio;
	std::vector<const char*> Songs;
	GW::AUDIO::GAudio audio;
	GW::AUDIO::GAudio vxfaudio;

	GW::INPUT::GInput keyboard;
	std::chrono::steady_clock::time_point now;

	int SongChoice = 0;
	int CurrentLevel = 0;
	// what we need at a minimum to draw a triangle
	VkDevice device = nullptr;
	VkBuffer vertexHandle = nullptr;
	VkDeviceMemory vertexData = nullptr;
	// TODO: Part 1g
	VkBuffer vertexHandle2 = nullptr;
	VkDeviceMemory vertexData2 = nullptr;
	// TODO: Part 2c
	std::vector<VkBuffer> uniformHandle;
	std::vector<VkDeviceMemory> uniformData;
	VkShaderModule vertexShader = nullptr;
	VkShaderModule pixelShader = nullptr;
	// pipeline settings for drawing (also required).
	VkPipeline pipeline = nullptr;
	VkPipelineLayout pipelineLayout = nullptr;
	// TODO: Part 2e
	VkDescriptorSetLayout descriptorLayout = nullptr;
	// TODO: Part 2f
	VkDescriptorPool descriptorPool = nullptr;
	//std::vector < VkDescriptorPool> descriptorPool;
	// TODO: Part 2g
	std::vector<VkDescriptorSet> descriptorSet;
	
	// descriptorset layout for vertex shader so we don't need to bind the texture
	VkDescriptorSetLayout vertexDescriptorLayout = nullptr;

	/***************** KTX+VULKAN TEXTURING VARIABLES ******************/

		// what we need at minimum to load/apply one texture
	ktxVulkanTexture texture; // one per texture
	VkImageView textureView = nullptr; // one per texture
	VkSampler textureSampler = nullptr; // can be shared, effects quality & addressing mode
	// note that unlike uniform buffers, we don't need one for each "in-flight" frame
	VkDescriptorSet textureDescriptorSet = nullptr; // std::vector<> not required

	// be aware that all pipeline shaders share the same bind points
	VkDescriptorSetLayout pixelDescriptorLayout = nullptr;
	// textures can optionally share descriptor sets/pools/layouts with uniform & storage buffers	
	//VkDescriptorPool descriptorPool = nullptr;
	// pipeline settings for drawing (also required)
	//VkPipeline pipeline = nullptr;
	//VkPipelineLayout pipelineLayout = nullptr;

	std::chrono::steady_clock::time_point tickTime;
	
	GW::MATH::GMatrix proxy;
	GW::MATH::GVector VectorProxy;

	GW::MATH::GMATRIXF GMatrixW;
	GW::MATH::GMATRIXF GMatrixV;
	GW::MATH::GMATRIXF ProjectionMatrix;
	GW::MATH::GMATRIXF WorldCamera;

	GW::MATH::GVECTORF LightDirectionVector;
	GW::MATH::GVECTORF LightColorVector;

	VkPhysicalDevice physicalDevice = nullptr;
	//texture and Skybox
	uint32_t indexCount;
	VkBuffer vertexHandleT = nullptr;
	VkDeviceMemory vertexDataT = nullptr;
	VkShaderModule vertexShaderT = nullptr;
	VkShaderModule pixelShaderT = nullptr;

public:
	void GEOMETRY_INTIALIZATION()
	{
		// Grab the device & physical device so we can allocate some stuff
		physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);

		H2B::VERTEX* tempVec = &ModelData.Vertexes[0];
		GvkHelper::create_buffer(physicalDevice, device, sizeof(*tempVec) * ModelData.Vertexes.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexHandle, &vertexData);
		GvkHelper::write_to_buffer(device, vertexData, &*tempVec, sizeof(*tempVec) * ModelData.Vertexes.size());
		// TODO: Part 1g
		unsigned* tempInd = &ModelData.Indexes[0];
		GvkHelper::create_buffer(physicalDevice, device, sizeof(*tempInd) * ModelData.Indexes.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexHandle2, &vertexData2);
		GvkHelper::write_to_buffer(device, vertexData2, &*tempInd, sizeof(*tempInd) * ModelData.Indexes.size());
	}
	void PopulateModel(std::map<std::string, Model_Struct> MeshMap)
	{
		int counterMat = 0;
		int counterV = 0;
		int counterMatx = 0;
		int counterI = 0;
		ModelData.MaterialOffset.push_back(0);
		ModelData.VertexOffsets.push_back(0);
		ModelData.MatrixOffset.push_back(0);
		ModelData.IndexOffsets.push_back(0);
		for (auto const& it : MeshMap)
		{

			//counter sizes
			counterMat += it.second.h2bParser.materialCount;
			counterV += it.second.h2bParser.vertices.size();
			counterMatx += it.second.WorldMatrices.size();
			counterI += it.second.h2bParser.indices.size();
			//make offsets
			ModelData.MaterialOffset.push_back(counterMat);
			ModelData.VertexOffsets.push_back(counterV);
			ModelData.MatrixOffset.push_back(counterMatx);
			ModelData.IndexOffsets.push_back(counterI);
			//pass V & I
			ModelData.Vertexes.insert(ModelData.Vertexes.end(), it.second.h2bParser.vertices.begin(), it.second.h2bParser.vertices.end());
			ModelData.Indexes.insert(ModelData.Indexes.end(), it.second.h2bParser.indices.begin(), it.second.h2bParser.indices.end());

		}
		//sizeof(std::vector<int>) + (sizeof(int) * MyVector.size())
		ModelData.VertexSize = sizeof(std::vector<H2B::VECTOR>) + (sizeof(H2B::VECTOR) * ModelData.Vertexes.size());
		ModelData.IndexSize = sizeof(ModelData.Indexes) * ModelData.Indexes.size();
	}
	bool ParseFile(const char* file) {
		int i, y, j;
		std::string line;
		std::ifstream myfile(file);
		std::string delimiter;
		std::string modelName;
		GW::MATH::GMATRIXF matrixTranfrom = GW::MATH::GIdentityMatrixF;
		if (!myfile.is_open()) {
			std::cout << "Unable to open file";
			return false;
		}
		while (std::getline(myfile, line)) {
			
			if (line == "MESH") {
				i = 0, y = 0, j = 0;
				std::getline(myfile, line);
				delimiter = ".";
				modelName = line.substr(0, line.find(delimiter));
				while (y < 4)
				{
					std::getline(myfile, line);
					delimiter = ")";
					size_t pos = 0;
					std::string sentence;
					line = line.substr(13, (line.find(delimiter) - 13));
					delimiter = ",";
					/*for (int o = 0; o < 4; o++) {
						pos = line.find(delimiter);
						sentence = line.substr(0, pos);
						matrixTranfrom.data[i] = std::stof(sentence);
						line.erase(0, pos + delimiter.length());
						i++;
					}*/
					//Line 1
					pos = line.find(delimiter);
					sentence = line.substr(0, pos);
					matrixTranfrom.data[i] = std::stof(sentence);
					line.erase(0, pos + delimiter.length());
					i++;
					//Line 1
					pos = line.find(delimiter);
					sentence = line.substr(0, pos);
					matrixTranfrom.data[i] = std::stof(sentence);
					line.erase(0, pos + delimiter.length());
					i++;
					//Line 1
					pos = line.find(delimiter);
					sentence = line.substr(0, pos);
					matrixTranfrom.data[i] = std::stof(sentence);
					line.erase(0, pos + delimiter.length());
					i++;
					//Line 1
					pos = line.find(delimiter);
					sentence = line.substr(0, pos);
					matrixTranfrom.data[i] = std::stof(sentence);
					line.erase(0, pos + delimiter.length());
					i++;

					y++;
				}
				if (mapModels.find(modelName) == mapModels.end()) {
					std::string filePath = "../../Assests/";
					filePath.append(modelName);
					filePath.append(".h2b");
					Model_Struct transformStruct = { };
					transformStruct.filename = modelName;
					transformStruct.filepath = filePath;
					transformStruct.WorldMatrices = { matrixTranfrom };
					//transformStruct.h2bParser = filePath;
					mapModels.insert(std::pair<std::string, Model_Struct>(modelName, transformStruct));
					mapModels[modelName].h2bParser.Parse(filePath.c_str());
				}
				else
				{
					mapModels[modelName].WorldMatrices.push_back(matrixTranfrom);
				}
			}
		}

		PopulateModel(mapModels);

		myfile.close();
		return true;
	}
	void Music()
	{
		audio.Create();
		music.Create("../ambience_city.wav", audio, 0.1);
		audio.PlayMusic();
		
	}
	
	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GVulkanSurface _vlk)
	{
		win = _win;
		vlk = _vlk;
		proxy.Create();
		VectorProxy.Create();
		keyboard.Create(win);
		now = std::chrono::steady_clock::now();
		//loading Music
		Music();
		
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		
		//loading Level
		ParseFile("../GameLevel2.txt");

		GW::MATH::GVECTORF vector = { 0.15f, 0.75f, 0, 0 };
		proxy.IdentityF(GMatrixW);
		proxy.RotateXGlobalF(GMatrixW, G2D_DEGREE_TO_RADIAN(90.0), GMatrixW);
		proxy.TranslateGlobalF(GMatrixW, vector, GMatrixW);

		//
		proxy.IdentityF(GMatrixV);
		//view
		GW::MATH::GVECTORF YUp = { 0, 1.0f, 0, 0 };
		GW::MATH::GVECTORF YAtCam = vector;
		GW::MATH::GVECTORF YEyeVTrans = { 9.75f, 4.25f, -10.5f, 0 };
		//proxy.TranslateGlobalF(GMatrixV, YEyeVTrans, GMatrixV);
		proxy.LookAtLHF(YEyeVTrans, YAtCam, YUp, GMatrixV);
		
		proxy.InverseF(GMatrixV, WorldCamera);
		
		//Projection
		float AspectRatio;
		proxy.IdentityF(ProjectionMatrix);
		//proxy.IdentityF(GMatrixP);
		vlk.GetAspectRatio(AspectRatio);
		GW::MATH::GVECTORF NP;
		GW::MATH::GVECTORF FP;
		proxy.ProjectionVulkanLHF(G2D_DEGREE_TO_RADIAN(65.0), AspectRatio, 0.1f, 100.0f, ProjectionMatrix);
		//
		LightDirectionVector = { -1.0f, -1.0f, 1.0f, 0.0f };
		VectorProxy.NormalizeF(LightDirectionVector, LightDirectionVector);
		int lightCount = 2;
		LIGHT lightValues[MAX_SUBMESH_PER_DRAW] =
		{
			{ 1,8, { -1,10,5,5 }, { 0.9f, 0.9f, 1.f, 0.7f }, LightDirectionVector },
			{ 2,2, { -1,2,0,0 }, { 0.0f, 0.0f, 0.1f, 0.01f },  { -1.0f, -1.0f, 1.0f, 0.0f } }
		};

		modelIndex.Light_ID = 0;
		//for (size_t i = 0; i < lightCount; i++)
		//{
		//	//ligth
		//	ShaderModelData.light[i] = lightValues[i];
		//	//modelIndex.Light_ID = i;
		//}
	
		////ligth
		ShaderModelData.light[0].radius = 8;
		ShaderModelData.light[0].color = { 0.9f, 0.9f, 1.f, 0.7f };
		ShaderModelData.light[0].pos = { -1,10,5,5 };
		ShaderModelData.light[0].dir = LightDirectionVector;
		ShaderModelData.light[1].radius = 2;
		ShaderModelData.light[1].color = { 0.2f, 0.2f, 1.f, 0.7f };
		ShaderModelData.light[1].pos = { -1,10,5,5 };
		ShaderModelData.light[1].dir = LightDirectionVector;
		LightColorVector = { 0.9f, 0.9f, 1.0f, 1.0f };

		// TODO: Part 2b
		ShaderModelData.matricies[0] = GMatrixW;
		ShaderModelData.ViewMatrix = GMatrixV;
		ShaderModelData.camPos = { 0.75f, 0.25f, -1.5f, 1 };
		ShaderModelData.ProjectionMatrix = ProjectionMatrix;
		ShaderModelData.sunAmbient = { 0.25f, 0.25f, 0.35f, 1 };
		ShaderModelData.sunColor = LightColorVector;
		ShaderModelData.sunDirection = LightDirectionVector;
		
		//W & Mat
		int matId = 0;
		int matxId = 0;
		for (auto const& it : mapModels)
		{
			for (int y = 0; y < it.second.h2bParser.materialCount; y++)
			{
				ShaderModelData.materials[matId] = it.second.h2bParser.materials[y].attrib;
				matId++;
			}
			for (int y = 0; y < it.second.WorldMatrices.size(); y++) 
			{
				ShaderModelData.matricies[matxId] = it.second.WorldMatrices[y];
				matxId++;
			}
		}

		/***************** GEOMETRY INTIALIZATION ******************/
		GEOMETRY_INTIALIZATION();

		// TODO: Part 2d
		unsigned max_frames = 0;
		// to avoid per-frame resource sharing issues we give each "in-flight" frame its own buffer
		vlk.GetSwapchainImageCount(max_frames);
		uniformHandle.resize(max_frames);
		uniformData.resize(max_frames);
		for (int i = 0; i < max_frames; ++i) {

			GvkHelper::create_buffer(physicalDevice, device, sizeof(ShaderModelData),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &uniformHandle[i], &uniformData[i]);
			GvkHelper::write_to_buffer(device, uniformData[i], &ShaderModelData, sizeof(ShaderModelData));
		} 
		/***************** SHADER INTIALIZATION ******************/
		// Intialize runtime shader compiler HLSL -> SPIRV
		shaderc_compiler_t compiler = shaderc_compiler_initialize();
		shaderc_compile_options_t options = shaderc_compile_options_initialize();
		shaderc_compile_options_set_source_language(options, shaderc_source_language_hlsl);
		shaderc_compile_options_set_invert_y(options, false); // TODO: Part 2i
#ifndef NDEBUG
		shaderc_compile_options_set_generate_debug_info(options);
#endif
		// Create Vertex Shader
		shaderc_compilation_result_t result = shaderc_compile_into_spv( // compile
			compiler, vertexShaderSource, strlen(vertexShaderSource),
			shaderc_vertex_shader, "main.vert", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Vertex Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &vertexShader);
		shaderc_result_release(result); // done
		// Create Pixel Shader
		result = shaderc_compile_into_spv( // compile
			compiler, pixelShaderSource, strlen(pixelShaderSource),
			shaderc_fragment_shader, "main.frag", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Pixel Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &pixelShader);
		shaderc_result_release(result); // done
		// Free runtime shader compiler resources
		shaderc_compile_options_release(options);
		shaderc_compiler_release(compiler);

#ifndef PIPELINE_INTIALIZATION
		/***************** PIPELINE INTIALIZATION ******************/
		// Create Pipeline & Layout (Thanks Tiny!)
		VkRenderPass renderPass;
		vlk.GetRenderPass((void**)&renderPass);
		VkPipelineShaderStageCreateInfo stage_create_info[2] = {};
		// Create Stage Info for Vertex Shader
		stage_create_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stage_create_info[0].module = vertexShader;
		stage_create_info[0].pName = "main";
		// Create Stage Info for Fragment Shader
		stage_create_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stage_create_info[1].module = pixelShader;
		stage_create_info[1].pName = "main";
		// Assembly State
		VkPipelineInputAssemblyStateCreateInfo assembly_create_info = {};
		assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assembly_create_info.primitiveRestartEnable = false;
		// TODO: Part 1e
		// Vertex Input State
		VkVertexInputBindingDescription vertex_binding_description = {};
		vertex_binding_description.binding = 0;
		vertex_binding_description.stride = sizeof(H2B::VERTEX);
		vertex_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription vertex_attribute_description[3] = {
			{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 }, //uv, normal, etc.... 
			{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12 },
			{ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24 }
		};
		VkPipelineVertexInputStateCreateInfo input_vertex_info = {};
		input_vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		input_vertex_info.vertexBindingDescriptionCount = 1;
		input_vertex_info.pVertexBindingDescriptions = &vertex_binding_description;
		input_vertex_info.vertexAttributeDescriptionCount = 3;
		input_vertex_info.pVertexAttributeDescriptions = vertex_attribute_description;
		// Viewport State (we still need to set this up even though we will overwrite the values)
		VkViewport viewport = {
			0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
		};
		VkRect2D scissor = { {0, 0}, {width, height} };
		VkPipelineViewportStateCreateInfo viewport_create_info = {};
		viewport_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_create_info.viewportCount = 1;
		viewport_create_info.pViewports = &viewport;
		viewport_create_info.scissorCount = 1;
		viewport_create_info.pScissors = &scissor;
		// Rasterizer State
		VkPipelineRasterizationStateCreateInfo rasterization_create_info = {};
		rasterization_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization_create_info.lineWidth = 1.0f;
		rasterization_create_info.cullMode = VK_CULL_MODE_NONE;
		rasterization_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterization_create_info.depthClampEnable = VK_FALSE;
		rasterization_create_info.depthBiasEnable = VK_FALSE;
		rasterization_create_info.depthBiasClamp = 0.0f;
		rasterization_create_info.depthBiasConstantFactor = 0.0f;
		rasterization_create_info.depthBiasSlopeFactor = 0.0f;
		// Multisampling State
		VkPipelineMultisampleStateCreateInfo multisample_create_info = {};
		multisample_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_create_info.sampleShadingEnable = VK_FALSE;
		multisample_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_create_info.minSampleShading = 1.0f;
		multisample_create_info.pSampleMask = VK_NULL_HANDLE;
		multisample_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_create_info.alphaToOneEnable = VK_FALSE;
		// Depth-Stencil State
		VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = {};
		depth_stencil_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil_create_info.depthTestEnable = VK_TRUE;
		depth_stencil_create_info.depthWriteEnable = VK_TRUE;
		depth_stencil_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_create_info.minDepthBounds = 0.0f;
		depth_stencil_create_info.maxDepthBounds = 1.0f;
		depth_stencil_create_info.stencilTestEnable = VK_FALSE;
		// Color Blending Attachment & State
		VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
		color_blend_attachment_state.colorWriteMask = 0xF;
		color_blend_attachment_state.blendEnable = VK_FALSE;
		color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
		color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
		color_blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		color_blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo color_blend_create_info = {};
		color_blend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_create_info.logicOpEnable = VK_FALSE;
		color_blend_create_info.logicOp = VK_LOGIC_OP_COPY;
		color_blend_create_info.attachmentCount = 1;
		color_blend_create_info.pAttachments = &color_blend_attachment_state;
		color_blend_create_info.blendConstants[0] = 0.0f;
		color_blend_create_info.blendConstants[1] = 0.0f;
		color_blend_create_info.blendConstants[2] = 0.0f;
		color_blend_create_info.blendConstants[3] = 0.0f;
		// Dynamic State 
		VkDynamicState dynamic_state[2] = {
			// By setting these we do not need to re-create the pipeline on Resize
			VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamic_create_info = {};
		dynamic_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_create_info.dynamicStateCount = 2;
		dynamic_create_info.pDynamicStates = dynamic_state;

		/***************** DESCRIPTOR SETUP FOR VERTEX & FRAGMENT SHADERS ******************/

		VkDescriptorSetLayoutBinding descriptor_layout_binding = {};
		descriptor_layout_binding.binding = 0;
		descriptor_layout_binding.descriptorCount = 1;
		descriptor_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		// In this scenario we have the same descriptorSetLayout for both shaders...
		// However, many times you would want seperate layouts for each since they tend to have different needs 
		descriptor_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		descriptor_layout_binding.pImmutableSamplers = nullptr;
		VkDescriptorSetLayoutCreateInfo descriptor_create_info = {};
		descriptor_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptor_create_info.flags = 0;
		descriptor_create_info.bindingCount = 1;
		descriptor_create_info.pBindings = &descriptor_layout_binding;
		descriptor_create_info.pNext = nullptr;
		// Descriptor layout
		VkResult r = vkCreateDescriptorSetLayout(device, &descriptor_create_info,
			nullptr, &descriptorLayout);
		// Create a descriptor pool!
		VkDescriptorPoolCreateInfo descriptorpool_create_info = {};
		descriptorpool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		VkDescriptorPoolSize descriptorpool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_frames };
		descriptorpool_create_info.poolSizeCount = 1;
		descriptorpool_create_info.pPoolSizes = &descriptorpool_size;
		descriptorpool_create_info.maxSets = max_frames;
		descriptorpool_create_info.flags = 0;
		descriptorpool_create_info.pNext = nullptr;
		vkCreateDescriptorPool(device, &descriptorpool_create_info, nullptr, &descriptorPool);
		// Create a descriptorSet for each uniform buffer!
		VkDescriptorSetAllocateInfo descriptorset_allocate_info = {};
		descriptorset_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descriptorset_allocate_info.descriptorSetCount = 1;
		descriptorset_allocate_info.pSetLayouts = &descriptorLayout;
		descriptorset_allocate_info.descriptorPool = descriptorPool;
		descriptorset_allocate_info.pNext = nullptr;
		descriptorSet.resize(max_frames);
		for (int i = 0; i < max_frames; ++i) {
			vkAllocateDescriptorSets(device, &descriptorset_allocate_info, &descriptorSet[i]);
		}
		// link our descriptor sets to our uniform buffers (one for each bufferimage)
		// you can do this later on too for switching buffers, just don't expect rendering frames to wait
		VkWriteDescriptorSet write_descriptorset = {};
		write_descriptorset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptorset.descriptorCount = 1;
		write_descriptorset.dstArrayElement = 0;
		write_descriptorset.dstBinding = 0;
		write_descriptorset.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		for (int i = 0; i < max_frames; ++i) {
			write_descriptorset.dstSet = descriptorSet[i];
			VkDescriptorBufferInfo dbinfo = { uniformHandle[i], 0, VK_WHOLE_SIZE };
			write_descriptorset.pBufferInfo = &dbinfo;
			vkUpdateDescriptorSets(device, 1, &write_descriptorset, 0, nullptr);
		}

		VkPushConstantRange PCR2 = {};
		PCR2.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		//PCR2.offset = 0;
		PCR2.size = sizeof(Model_INDEX);

		// Descriptor pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &descriptorLayout;
		pipeline_layout_create_info.pushConstantRangeCount = 1;
		pipeline_layout_create_info.pPushConstantRanges = &PCR2;
		vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipelineLayout);

		VkGraphicsPipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = stage_create_info;
		pipeline_create_info.pInputAssemblyState = &assembly_create_info;
		pipeline_create_info.pVertexInputState = &input_vertex_info;
		pipeline_create_info.pViewportState = &viewport_create_info;
		pipeline_create_info.pRasterizationState = &rasterization_create_info;
		pipeline_create_info.pMultisampleState = &multisample_create_info;
		pipeline_create_info.pDepthStencilState = &depth_stencil_create_info;
		pipeline_create_info.pColorBlendState = &color_blend_create_info;
		pipeline_create_info.pDynamicState = &dynamic_create_info;
		pipeline_create_info.layout = pipelineLayout;
		pipeline_create_info.renderPass = renderPass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
			&pipeline_create_info, nullptr, &pipeline);
#endif

		// With pipeline created, lets load in our texture and bind it to our descriptor set
		//LoadTextures("../../Textures/greendragon.ktx");

		/***************** CLEANUP / SHUTDOWN ******************/
		// GVulkanSurface will inform us when to release any allocated resources
		shutdown.Create(vlk, [&]() {
			if (+shutdown.Find(GW::GRAPHICS::GVulkanSurface::Events::RELEASE_RESOURCES, true)) {
				CleanUp(); // unlike D3D we must be careful about destroy timing
			}
		});
	}

	/***************** KTX TEXTURE LOADING & VULKAN SAMPLER/IMGVIEW CREATION ******************/

	// ideally this would take multiple/all textures you want to load
	bool LoadTextures(const char* texturePath)
	{
		// Gateware, access to underlying Vulkan queue and command pool & physical device
		VkQueue graphicsQueue;
		VkCommandPool cmdPool;
		VkPhysicalDevice physicalDevice;
		vlk.GetGraphicsQueue((void**)&graphicsQueue);
		vlk.GetCommandPool((void**)&cmdPool);
		vlk.GetPhysicalDevice((void**)&physicalDevice);
		// libktx, temporary variables
		ktxTexture* kTexture;
		KTX_error_code ktxresult;
		ktxVulkanDeviceInfo vdi;
		// used to transfer texture CPU memory to GPU. just need one
		ktxresult = ktxVulkanDeviceInfo_Construct(&vdi, physicalDevice, device,
			graphicsQueue, cmdPool, nullptr);
		if (ktxresult != KTX_error_code::KTX_SUCCESS)
			return false;
		// load texture into CPU memory from file
		ktxresult = ktxTexture_CreateFromNamedFile(texturePath,
			KTX_TEXTURE_CREATE_NO_FLAGS, &kTexture);
		if (ktxresult != KTX_error_code::KTX_SUCCESS)
			return false;
		// This gets mad if you don't encode/save the .ktx file in a format Vulkan likes
		ktxresult = ktxTexture_VkUploadEx(kTexture, &vdi, &texture,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		if (ktxresult != KTX_error_code::KTX_SUCCESS)
			return false;
		// after loading all textures you don't need these anymore
		ktxTexture_Destroy(kTexture);
		ktxVulkanDeviceInfo_Destruct(&vdi);

		// create the the image view and sampler
		VkSamplerCreateInfo samplerInfo = {};
		// Set the struct values
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.flags = 0;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; // REPEAT IS COMMON
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0;
		samplerInfo.minLod = 0;
		samplerInfo.maxLod = texture.levelCount;
		samplerInfo.anisotropyEnable = VK_FALSE;
		samplerInfo.maxAnisotropy = 1.0;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_LESS;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.pNext = nullptr;
		VkResult vr = vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler);
		if (vr != VkResult::VK_SUCCESS)
			return false;

		// Create image view.
		// Textures are not directly accessed by the shaders and are abstracted
		// by image views containing additional information and sub resource ranges.
		VkImageViewCreateInfo viewInfo = {};
		// Set the non-default values.
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.flags = 0;
		viewInfo.components = {
			VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
			VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A
		};
		viewInfo.image = texture.image;
		viewInfo.format = texture.imageFormat;
		viewInfo.viewType = texture.viewType;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.layerCount = texture.layerCount;
		viewInfo.subresourceRange.levelCount = texture.levelCount;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.pNext = nullptr;
		vr = vkCreateImageView(device, &viewInfo, nullptr, &textureView);
		if (vr != VkResult::VK_SUCCESS)
			return false;

		// update the descriptor set(s) to point to the correct views
		VkWriteDescriptorSet write_descriptorset = {};
		write_descriptorset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptorset.descriptorCount = 1;
		write_descriptorset.dstArrayElement = 0;
		write_descriptorset.dstBinding = 1;
		write_descriptorset.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		write_descriptorset.dstSet = textureDescriptorSet;
		VkDescriptorImageInfo diinfo = { textureSampler, textureView, texture.imageLayout };
		write_descriptorset.pImageInfo = &diinfo;
		vkUpdateDescriptorSets(device, 1, &write_descriptorset, 0, nullptr);

		return true;
	}

	void Render()
	{
		//Variables
		int offSet = 0;
		unsigned counter = 1;

		unsigned int currentBuffer;
		vlk.GetSwapchainCurrentImage(currentBuffer);
		VkCommandBuffer commandBuffer;
		vlk.GetCommandBuffer(currentBuffer, (void**)&commandBuffer);
		// what is the current client area dimensions?
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// setup the pipeline's dynamic settings
		VkViewport viewport = {
			0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
		};
		VkRect2D scissor = { {0, 0}, {width, height} };
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		// now we can draw
		VkDeviceSize offsets[] = { 0 };

		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexHandle, offsets);
		vkCmdBindIndexBuffer(commandBuffer, vertexHandle2, *offsets, VK_INDEX_TYPE_UINT32);

		GvkHelper::write_to_buffer(device, uniformData[currentBuffer], &ShaderModelData, sizeof(ShaderModelData));
		/***************** BINDING OF UNIFORM BUFFER VIA DESCRIPTORSET ******************/

		// *NEW* Set the descriptorSet that contains the uniform buffer allocated for this framebuffer 

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipelineLayout, 0, 1, &descriptorSet[currentBuffer], 0, nullptr);

		/*for (size_t i = 0; i < FSLogo_materialcount; i++)
		{
			vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(unsigned), &i);
			vkCmdDrawIndexed(commandBuffer, FSLogo_meshes[i].indexCount, 1, FSLogo_meshes[i].indexOffset, 0, 0);
		}*/

		/***************** BINDING OF TEXTURE DESCRIPTOR SET TO PIXEL SHADER ******************/

		// *NEW* Set the descriptorSet that contains our texture
		//vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			//pipelineLayout, 1, 1, &textureDescriptorSet, 0, nullptr);
		
			

		for (iter = mapModels.begin(); iter != mapModels.end(); ++iter, offSet++) 
		{
			for (unsigned i = 0; i < iter->second.h2bParser.meshCount; i++)
			{
			
				modelIndex.mesh_ID = ModelData.MaterialOffset[offSet] + iter->second.h2bParser.meshes[i].materialIndex;
				modelIndex.wm_ID = counter;
				//modelIndex.Light_ID = counter;
				
				vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Model_INDEX), &modelIndex);
				vkCmdDrawIndexed(commandBuffer, iter->second.h2bParser.meshes[i].drawInfo.indexCount, iter->second.WorldMatrices.size(), ModelData.IndexOffsets[offSet] + iter->second.h2bParser.meshes[i].drawInfo.indexOffset, ModelData.VertexOffsets[offSet], ModelData.MatrixOffset[offSet]);
				//vkCmdDrawIndexed(commandBuffer, it->second.h2bParser.meshes[i].drawInfo.indexCount, 1,  it->second.h2bParser.meshes[i].drawInfo.indexOffset, VertexOffsets[offSet], 0);
			}
		}
	}

	void clearUp()
	{
		vkDeviceWaitIdle(device);
		vkFreeMemory(device, vertexData, nullptr);
		vkDestroyBuffer(device, vertexHandle, nullptr);
		vkFreeMemory(device, vertexData2, nullptr);
		vkDestroyBuffer(device, vertexHandle2, nullptr);
		mapModels.clear();
		ModelData.Vertexes.clear();
		ModelData.Indexes.clear();
		ModelData.IndexOffsets.clear();
		ModelData.VertexOffsets.clear();
		ModelData.MaterialOffset.clear();
		ModelData.MatrixOffset.clear();
	}

	void changeLevel() {
		
		clearUp();

		if (CurrentLevel == 0) {
			ParseFile("../GameLevel2.txt");
			CurrentLevel = 1;
		}
		else {
			ParseFile("../GameLevel.txt");
			CurrentLevel = 0;
		}
		//W & Mat
		int matId = 0;
		int matxId = 0;
		for (auto const& it : mapModels)
		{
			for (int y = 0; y < it.second.h2bParser.materialCount; y++)
			{
				ShaderModelData.materials[matId] = it.second.h2bParser.materials[y].attrib;
				matId++;
			}
			for (int y = 0; y < it.second.WorldMatrices.size(); y++)
			{
				ShaderModelData.matricies[matxId] = it.second.WorldMatrices[y];
				matxId++;
			}
		}

		GEOMETRY_INTIALIZATION();
	}

	void Input()
	{
		static bool LevelChange = true;
		float R;
		keyboard.GetState(55, R);
		static bool LevelChanged;
		
		if (LevelChange && !LevelChanged) {
			LevelChange = false;
			changeLevel();
			LevelChanged = true;
		}
		if (R == 0)
		{
			LevelChange = false;
			LevelChanged = false;
		}	
		else 
		{
			LevelChange = true;
		}

		if (GetAsyncKeyState('F'))
		{
			vxfaudio.Create();
			vxfmusic.Create("../door_open.wav", vxfaudio, 0.5);
			vxfaudio.PlayMusic();
		}

	}

	void UpdateCamera()
	{

		proxy.InverseF(ShaderModelData.ViewMatrix, WorldCamera);

		//InputVariables
		float aspectratio = 0;
		unsigned height, width;
		vlk.GetAspectRatio(aspectratio);
		win.GetHeight(height);
		win.GetWidth(width);
		const float camera_speed = 1.0f;

		//Keybord Input
		float spacebar = 0;
		float leftShift = 0;
		keyboard.GetState(23, spacebar);
		keyboard.GetState(14, leftShift);
		float Y_Change = spacebar - leftShift;


		//Movement
		float A = 0;
		float W = 0;
		float S = 0;
		float D = 0;
		keyboard.GetState(60, W);
		keyboard.GetState(56, S);
		keyboard.GetState(41, D);
		keyboard.GetState(38, A);
		float totalZ = W - S;
		float totalX = D - A;

		//Mouse Input
		float MYAxis = 0;
		float MXAxis = 0;
		float total_pitch = 0;
		float total_yaw = 0;
		GW::GReturn returnValue = GW::GReturn::FAILURE;;
		returnValue = keyboard.GetMouseDelta(MXAxis, MYAxis);

		//Tick Time
		auto tickTimeEnd = std::chrono::steady_clock::now();
		std::chrono::duration<float> delta = tickTimeEnd - tickTime;
		tickTime = tickTimeEnd;
		float fps = camera_speed * delta.count() * 4;
		float Total_Y_Change = Y_Change * 500 * fps * delta.count();

		//Add Keyboard Movement
		GW::MATH::GVECTORF vectorGUpDown = { 0.0f,Total_Y_Change,0.0f,1.0f };
		GW::MATH::GVECTORF vectorMovement = { totalX * fps, 0, totalZ * fps };
		proxy.TranslateGlobalF(WorldCamera, vectorGUpDown, WorldCamera);
		proxy.TranslateLocalF(WorldCamera, vectorMovement, WorldCamera);

		//Add Mouse Movement
		if (returnValue == GW::GReturn::SUCCESS)
		{
			total_pitch = total_pitch + 1.13446f * MYAxis / height;
			total_yaw = total_yaw + 1.13446f * aspectratio * MXAxis / width;
		}
		GW::MATH::GMATRIXF pitch;
		GW::MATH::GMATRIXF yaw;
		proxy.IdentityF(pitch);
		proxy.RotateYGlobalF(WorldCamera, total_yaw, WorldCamera);
		//proxy.MultiplyMatrixF(pitch, WorldCamera, WorldCamera);
		proxy.IdentityF(yaw);
		proxy.RotateXLocalF(WorldCamera, total_pitch, WorldCamera);
		//proxy.MultiplyMatrixF(WorldCamera, yaw, WorldCamera);

		proxy.InverseF(WorldCamera, ShaderModelData.ViewMatrix);

	};
	
private:
	void CleanUp()
	{
		// wait till everything has completed
		vkDeviceWaitIdle(device);
		// Release allocated buffers, shaders & pipeline
		// // When done using the image in Vulkan...
		/*ktxVulkanTexture_Destruct(&texture, device, nullptr);
		if (textureView) {
			vkDestroyImageView(device, textureView, nullptr);
			textureView = nullptr;
		}
		if (textureSampler) {
			vkDestroySampler(device, textureSampler, nullptr);
			textureSampler = nullptr;
		}*/
		// TODO: Part 1g
		vkDestroyBuffer(device, vertexHandle2, nullptr);
		vkFreeMemory(device, vertexData2, nullptr);
		// TODO: Part 2d
		for (int i = 0; i < uniformHandle.size(); i++) {
			vkDestroyBuffer(device, uniformHandle[i], nullptr);
			vkFreeMemory(device, uniformData[i], nullptr);
			//vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		}
		uniformHandle.clear();
		uniformData.clear();
		vkDestroyBuffer(device, vertexHandle, nullptr);
		vkFreeMemory(device, vertexData, nullptr);
		vkDestroyShaderModule(device, vertexShader, nullptr);
		vkDestroyShaderModule(device, pixelShader, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
		// don't need the descriptors anymore
		vkDestroyDescriptorSetLayout(device, vertexDescriptorLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, pixelDescriptorLayout, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
	}
};
