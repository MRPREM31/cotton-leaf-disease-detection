package com.cottonguard.ai

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.cottonguard.ai.ml.CottonNetClassifier
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors

data class ScanHistoryItem(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: String,
    val className: String,
    val confidence: Float,
    val latencyMs: Long,
    val bitmap: Bitmap?
)

data class DiseaseDetails(
    val icon: String,
    val englishName: String,
    val symptoms: String,
    val chemicalControl: String,
    val organicControl: String,
    val severity: String,
    val severityColor: Color
)

class MainActivity : ComponentActivity() {

    private lateinit var classifier: CottonNetClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            classifier = CottonNetClassifier(this)
        } catch (e: Exception) {
            Log.e("CottonGuard", "Classifier load error", e)
        }

        setContent {
            CottonGuardTheme {
                CottonGuardApp(classifier = classifier)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::classifier.isInitialized) {
            classifier.close()
        }
    }
}

// ==================== AGRONOMIC KNOWLEDGE BASE ====================
val DISEASE_ADVISORY: Map<String, Map<String, DiseaseDetails>> = mapOf(
    "Bacterial Blight" to mapOf(
        "en" to DiseaseDetails(
            icon = "🦠",
            englishName = "Bacterial Blight (Xanthomonas malvacearum)",
            symptoms = "Water-soaked angular lesions delimited by leaf veins, darkening to reddish-brown/black. Spreads to branches causing 'black arm' and boll rot.",
            chemicalControl = "Spray Copper Oxychloride 50 WP (2.5 g/L) + Streptocycline (100 mg/L) upon early detection. Repeat after 15 days.",
            organicControl = "Apply 5% Neem Seed Kernel Extract (NSKE) or Pseudomonas fluorescens (10 g/L). Avoid excessive urea nitrogen.",
            severity = "Severe Infection",
            severityColor = Color(0xFFEF4444)
        ),
        "hi" to DiseaseDetails(
            icon = "🦠",
            englishName = "जीवाणु झुलसा (बैक्टीरियल ब्लाइट)",
            symptoms = "पत्तियों पर नसों से घिरे कोणीय काले-भूरे धब्बे। बाद में तनों पर काला घाव (ब्लैक आर्म) बनाता है जिससे गूलर गिर जाते हैं।",
            chemicalControl = "कॉपर ऑक्सीक्लोराइड 50 WP (2.5 ग्राम/लीटर) + स्ट्रेप्टोसाइक्लिन (100 मिलीग्राम/लीटर) का छिड़काव करें।",
            organicControl = "नीम बीज अर्क (NSKE 5%) या स्यूडोमोनास फ्लोरोसेंस (10 ग्राम/लीटर) का छिड़काव करें।",
            severity = "गंभीर संक्रमण",
            severityColor = Color(0xFFEF4444)
        ),
        "or" to DiseaseDetails(
            icon = "🦠",
            englishName = "ଜୀବାଣୁ ଜନିତ ପତ୍ରପୋଡ଼ା (Bacterial Blight)",
            symptoms = "ପତ୍ର ଶିରା ମଝିରେ କୋଣିଆ କଳା ଦାଗ ସୃଷ୍ଟି ହୁଏ । ଡାଳ କଳା ପଡ଼ିଯାଏ ଏବଂ କପା କଷି ଝଡ଼ିପଡ଼େ ।",
            chemicalControl = "କପର ଅକ୍ସିକ୍ଲୋରାଇଡ୍ 50 WP (୨.୫ ଗ୍ରାମ୍/ଲି) + ଷ୍ଟ୍ରେପ୍ଟୋସାଇକ୍ଲିନ୍ (୧୦୦ ମି.ଗ୍ରା/ଲି) ସ୍ପ୍ରେ କରନ୍ତୁ ।",
            organicControl = "ନିମ୍ବ ମଞ୍ଜି ଅର୍କ (୫%) କିମ୍ବା ସୁଡୋମୋନାସ (୧୦ ଗ୍ରାମ/ଲି) ପ୍ରୟୋଗ କରନ୍ତୁ ।",
            severity = "ଉଚ୍ଚ ସତର୍କତା",
            severityColor = Color(0xFFEF4444)
        )
    ),
    "Curl Virus" to mapOf(
        "en" to DiseaseDetails(
            icon = "🌀",
            englishName = "Cotton Leaf Curl Virus (CLCuV)",
            symptoms = "Upward/downward leaf curling, thickened veins, leafy outgrowths (enations) on underside, and severe plant stunting.",
            chemicalControl = "Suppress whitefly vectors: Spray Diafenthiuron 50 WP (1.0 g/L) or Flonicamid 50 WG (0.3 g/L).",
            organicControl = "Install yellow sticky traps (20/acre) and spray 5% neem oil emulsion. Rogue out infected plants early.",
            severity = "Viral Outbreak",
            severityColor = Color(0xFFF97316)
        ),
        "hi" to DiseaseDetails(
            icon = "🌀",
            englishName = "पत्ती मरोड़ वायरस (लीफ कर्ल वायरस)",
            symptoms = "पत्तियों के किनारों का ऊपर या नीचे मुड़ना, नसों का मोटा होना और पौधे का विकास रुक जाना।",
            chemicalControl = "सफेद मक्खी नियंत्रण हेतु डायफेन्थियूरॉन (1.0 ग्राम/लीटर) या फ्लोनिकैमिड (0.3 ग्राम/लीटर) छिड़कें।",
            organicControl = "पीले चिपचिपे ट्रैप (20 प्रति एकड़) लगाएं और 5% नीम तेल छिड़कें।",
            severity = "विषाणु संक्रमण",
            severityColor = Color(0xFFF97316)
        ),
        "or" to DiseaseDetails(
            icon = "🌀",
            englishName = "କପା ପତ୍ର କୁଞ୍ଚୁକୁଞ୍ଚିଆ ରୋଗ (Curl Virus)",
            symptoms = "ପତ୍ର ଉପରକୁ ବା ତଳକୁ ମୋଡ଼ି ହୋଇଯାଏ, ଶିରା ମୋଟା ହୁଏ ଏବଂ ଗଛର ବୃଦ୍ଧି ରହିଯାଏ ।",
            chemicalControl = "ଧଳାମାଛି ଦମନ ପାଇଁ ଡାୟାଫେନ୍ଥିୟୁରନ୍ (୧ ଗ୍ରାମ/ଲି) ବା ଫ୍ଲୋନିକାମିଡ୍ (୦.୩ ଗ୍ରାମ/ଲି) ସିଞ୍ଚନ କରନ୍ତୁ ।",
            organicControl = "ହଳଦିଆ ଷ୍ଟିକି ଫାନ୍ଦ (୨୦ଟି/ଏକର) ଲଗାନ୍ତୁ ଓ ୫% ନିମ୍ବ ତେଲ ସ୍ପ୍ରେ କରନ୍ତୁ ।",
            severity = "ଭାଇରସ୍ ଆକ୍ରମଣ",
            severityColor = Color(0xFFF97316)
        )
    ),
    "Healthy Leaf" to mapOf(
        "en" to DiseaseDetails(
            icon = "🌱",
            englishName = "Healthy Specimen (Optimal Health)",
            symptoms = "Uniform vibrant green leaf lamina, intact margins, active photosynthesis, absence of necrotic lesions or chlorosis.",
            chemicalControl = "No chemical pesticides required. Maintain balanced standard NPK nutrition (120:60:60 kg/ha).",
            organicControl = "Apply foliar sprays of Panchagavya (3%) or Jeevamrutha every 20 days to preserve plant vitality.",
            severity = "Healthy & Vigor",
            severityColor = Color(0xFF10B981)
        ),
        "hi" to DiseaseDetails(
            icon = "🌱",
            englishName = "स्वस्थ कपास पत्ती (रोगमुक्त)",
            symptoms = "पत्ता पूरी तरह हरा, साफ और किसी भी रोग या कीट क्षति से मुक्त है।",
            chemicalControl = "किसी रासायनिक कीटनाशक की आवश्यकता नहीं है। संतुलित पोषण बनाए रखें।",
            organicControl = "पौधे के अच्छे विकास के लिए पंचगव्य (3%) या जीवामृत का छिड़काव करें।",
            severity = "उत्कृष्ट स्थिति",
            severityColor = Color(0xFF10B981)
        ),
        "or" to DiseaseDetails(
            icon = "🌱",
            englishName = "ସୁସ୍ଥ କପା ପତ୍ର (ନିରୋଗ)",
            symptoms = "ପତ୍ର ସମ୍ପୂର୍ଣ୍ଣ ସବୁଜ, ନିରୋଗ ଓ ଉଜ୍ଜ୍ୱଳ । କୌଣସି ପୋକ ବା ରୋଗ ଦାଗ ନାହିଁ ।",
            chemicalControl = "କୌଣସି କୀଟନାଶକ ଆବଶ୍ୟକ ନାହିଁ । ସନ୍ତୁଳିତ ସାର ପ୍ରୟୋଗ କରନ୍ତୁ ।",
            organicControl = "ଜୀବାମୃତ ବା ପଞ୍ଚଗବ୍ୟ (୩%) ସ୍ପ୍ରେ କରି ଗଛକୁ ସତେଜ ରଖନ୍ତୁ ।",
            severity = "ଉତ୍ତମ ସ୍ୱାସ୍ଥ୍ୟ",
            severityColor = Color(0xFF10B981)
        )
    ),
    "Herbicide Growth Damage" to mapOf(
        "en" to DiseaseDetails(
            icon = "⚠️",
            englishName = "Herbicide Drift / Phytotoxicity",
            symptoms = "Strap-like narrowed leaf lobes (feathering), parallel venation chlorosis, and curled growing tips from 2,4-D or Glyphosate drift.",
            chemicalControl = "Foliar spray of 1% Urea + 1% Potassium Nitrate (10 g/L) + Zinc micronutrient to stimulate vegetative recovery.",
            organicControl = "Apply cold-water Seaweed Ascophyllum nodosum extract (@ 2 mL/L) to alleviate cellular phytotoxicity.",
            severity = "Abiotic Damage",
            severityColor = Color(0xFFEAB308)
        ),
        "hi" to DiseaseDetails(
            icon = "⚠️",
            englishName = "खरपतवारनाशक (हर्बिसाइड) क्षति",
            symptoms = "पत्तियों का पतला, लंबा खिंचना और विकृत होना (2,4-D या ग्लाइफोसेट के बहाव के कारण)।",
            chemicalControl = "1% यूरिया + 1% पोटेशियम नाइट्रेट और सूक्ष्म पोषक तत्वों का छिड़काव करें।",
            organicControl = "सीवीड अर्क (2 मिली/लीटर) का छिड़काव करके पौधे को तनावमुक्त करें।",
            severity = "रासायनिक तनाव",
            severityColor = Color(0xFFEAB308)
        ),
        "or" to DiseaseDetails(
            icon = "⚠️",
            englishName = "ଘାସମରା ଔଷଧ ଜନିତ କ୍ଷତି (Herbicide Damage)",
            symptoms = "ପତ୍ର ସରୁ, ଲମ୍ବା ହୋଇ ବିକୃତ ହୋଇଯାଏ ଓ ଶିରା ହଳଦିଆ ପଡ଼ିଯାଏ ।",
            chemicalControl = "୧% ୟୁରିଆ + ୧% ପୋଟାସିୟମ୍ ନାଇଟ୍ରେଟ୍ ଏବଂ ଜିଙ୍କ ଅଣୁସାର ସିଞ୍ଚନ କରନ୍ତୁ ।",
            organicControl = "ସାମୁଦ୍ରିକ ଶୈବାଳ ଅର୍କ (୨ ମି.ଲି/ଲି) ସ୍ପ୍ରେ କରି ଗଛକୁ ସଜାଡ଼ନ୍ତୁ ।",
            severity = "ଔଷଧୀୟ ଚାପ",
            severityColor = Color(0xFFEAB308)
        )
    ),
    "Leaf Hopper Jassids" to mapOf(
        "en" to DiseaseDetails(
            icon = "🦗",
            englishName = "Leaf Hopper / Jassids (Amrasca biguttula)",
            symptoms = "Downward curling of margins, peripheral chlorosis progressing to triangular brick-red hopper burn necrosis.",
            chemicalControl = "Spray Flonicamid 50 WG (0.3 g/L) or Dinotefuran 20 SG (0.4 g/L) or Imidacloprid 17.8 SL (0.3 mL/L).",
            organicControl = "Spray Beauveria bassiana 1.15% WP (5 g/L) in late afternoon with high relative humidity.",
            severity = "Pest Pressure",
            severityColor = Color(0xFFF59E0B)
        ),
        "hi" to DiseaseDetails(
            icon = "🦗",
            englishName = "हरा तेला / जैसिड्स (हॉपर बर्न)",
            symptoms = "पत्तियों के किनारों का नीचे मुड़ना, किनारों पर पीलापन और बाद में लाल होकर सूखना।",
            chemicalControl = "फ्लोनिकैमिड 50 WG (0.3 ग्राम/लीटर) या डाइनोटेफ्यूरॉन (0.4 ग्राम/लीटर) का छिड़काव करें।",
            organicControl = "ब्युवेरिया बासियाना (5 ग्राम/लीटर) जैविक कीटनाशक का छिड़काव करें।",
            severity = "कीट प्रकोप",
            severityColor = Color(0xFFF59E0B)
        ),
        "or" to DiseaseDetails(
            icon = "🦗",
            englishName = "ଡିଆଁ ପୋକ / ଜାସିଡ୍ (Leaf Hopper)",
            symptoms = "ପତ୍ର କଡ଼ ହଳଦିଆ ଓ ନାଲି ହୋଇ ତଳକୁ ମୋଡ଼ି ହୁଏ ଏବଂ ପତ୍ର ଶୁଖିଯାଏ ।",
            chemicalControl = "ଫ୍ଲୋନିକାମିଡ୍ 50 WG (୦.୩ ଗ୍ରାମ/ଲି) ବା ଡାଇନୋଟେଫ୍ୟୁରାନ୍ (୦.୪ ଗ୍ରାମ/ଲି) ସିଞ୍ଚନ କରନ୍ତୁ ।",
            organicControl = "ଜୈବିକ କବକ ବିଭେରିଆ ବାସିଆନା (୫ ଗ୍ରାମ/ଲି) ପ୍ରୟୋଗ କରନ୍ତୁ ।",
            severity = "ପୋକ ଆକ୍ରମଣ",
            severityColor = Color(0xFFF59E0B)
        )
    ),
    "Leaf Redding" to mapOf(
        "en" to DiseaseDetails(
            icon = "🍂",
            englishName = "Physiological Leaf Redding (Mg Deficiency)",
            symptoms = "Anthocyanin accumulation turning leaves wine-red to purple during boll maturation while veins remain green.",
            chemicalControl = "Foliar spray of 2% Magnesium Sulfate (MgSO4 - 20 g/L) + 1% Urea at 15-day intervals during boll development.",
            organicControl = "Ensure field drainage to eliminate root hypoxia. Apply well-decomposed FYM with dolomite lime.",
            severity = "Nutrient Deficit",
            severityColor = Color(0xFFA855F7)
        ),
        "hi" to DiseaseDetails(
            icon = "🍂",
            englishName = "पत्ती लाल होना (मैग्नीशियम कमी)",
            symptoms = "गूलर बनते समय पत्तियों का नसों के बीच से लाल-बैंगनी होना। नसों का हरा रहना।",
            chemicalControl = "2% मैग्नीशियम सल्फेट (20 ग्राम/लीटर) + 1% यूरिया का छिड़काव करें।",
            organicControl = "खेत में जलभराव रोकें और मिट्टी में गोबर की पकी खाद का प्रयोग करें।",
            severity = "पोषक तत्व कमी",
            severityColor = Color(0xFFA855F7)
        ),
        "or" to DiseaseDetails(
            icon = "🍂",
            englishName = "କପା ପତ୍ର ନାଲି ପଡ଼ିବା (Leaf Redding)",
            symptoms = "ମ୍ୟାଗ୍ନେସିୟମ୍ ଅଭାବ ଓ ଥଣ୍ଡା ଯୋଗୁଁ କପା ଫଳ ହେବା ବେଳେ ପତ୍ର ନାଲି-ବାଇଗଣି ରଙ୍ଗ ଧରେ ।",
            chemicalControl = "୨% ମ୍ୟାଗ୍ନେସିୟମ୍ ସଲଫେଟ୍ (୨୦ ଗ୍ରାମ/ଲି) + ୧% ୟୁରିଆ ମିଶାଇ ସ୍ପ୍ରେ କରନ୍ତୁ ।",
            organicControl = "ଜମିରୁ ଜଳ ନିଷ୍କାସନ ଠିକ୍ କରନ୍ତୁ ଏବଂ କମ୍ପୋଷ୍ଟ ଖତ ବ୍ୟବହାର କରନ୍ତୁ ।",
            severity = "ପୋଷକ ଅଭାବ",
            severityColor = Color(0xFFA855F7)
        )
    ),
    "Leaf Variegation" to mapOf(
        "en" to DiseaseDetails(
            icon = "🎨",
            englishName = "Micronutrient / Genetic Variegation",
            symptoms = "Irregular yellow, ivory, and emerald green mosaic patches across leaf lamina without necrotic lesions.",
            chemicalControl = "Foliar spray of Chelated Zinc (1.0 g/L) + Ferrous Sulfate (2.0 g/L) + Boron (1.0 g/L).",
            organicControl = "Soil incorporation of mycorrhizal bio-fertilizers and compost enriched with Trichoderma.",
            severity = "Minor Condition",
            severityColor = Color(0xFF06B6D4)
        ),
        "hi" to DiseaseDetails(
            icon = "🎨",
            englishName = "पत्ती चितकबरापन (सूक्ष्मपोषक कमी)",
            symptoms = "पत्ती की सतह पर अनियमित पीले, सफेद और हरे धब्बे।",
            chemicalControl = "चिलेटेड जिंक (1 ग्राम/लीटर) + फेरस सल्फेट (2 ग्राम/लीटर) का छिड़काव करें।",
            organicControl = "माइकोराइजा युक्त जैविक खाद मिट्टी में मिलाएं।",
            severity = "सूक्ष्मपोषक कमी",
            severityColor = Color(0xFF06B6D4)
        ),
        "or" to DiseaseDetails(
            icon = "🎨",
            englishName = "ପତ୍ର ଚିତ୍ରବିଚିତ୍ର ଦାଗ (Leaf Variegation)",
            symptoms = "ପତ୍ର ଉପରେ ହଳଦିଆ, ଧଳା ଓ ସବୁଜ ମିଶାମିଶି ରଙ୍ଗର ଦାଗ ଦେଖାଯାଏ ।",
            chemicalControl = "ଜିଙ୍କ ଅଣୁସାର (୧ ଗ୍ରାମ/ଲି) + ଆଇରନ ସଲଫେଟ୍ (୨ ଗ୍ରାମ/ଲି) ସ୍ପ୍ରେ କରନ୍ତୁ ।",
            organicControl = "ଜୈବିକ ଖତ ଓ ମାଇକୋରାଇଜା ମାଟିରେ ପ୍ରୟୋଗ କରନ୍ତୁ ।",
            severity = "ସାମାନ୍ୟ ଅସନ୍ତୁଳନ",
            severityColor = Color(0xFF06B6D4)
        )
    )
)

// Helper to safely decode Bitmaps from URIs
fun loadBitmapFromUri(context: Context, uri: Uri): Bitmap? {
    return try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }
    } catch (e: Exception) {
        Log.e("CottonGuard", "Failed to load URI bitmap", e)
        null
    }
}

// Helper to safely load Sample Bitmaps from Assets
fun loadBitmapFromAsset(context: Context, assetPath: String): Bitmap? {
    return try {
        context.assets.open(assetPath).use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        }
    } catch (e: Exception) {
        Log.e("CottonGuard", "Failed to load Asset bitmap", e)
        null
    }
}

// ==================== MAIN COMPOSABLE ====================
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CottonGuardApp(classifier: CottonNetClassifier) {
    val context = LocalContext.current
    var selectedTab by remember { mutableStateOf(0) }
    var selectedLanguage by remember { mutableStateOf("en") }
    var diagnosisResult by remember { mutableStateOf<CottonNetClassifier.Recognition?>(null) }
    var selectedImage by remember { mutableStateOf<Bitmap?>(null) }
    val historyList = remember { mutableStateListOf<ScanHistoryItem>() }
    var isAnalyzing by remember { mutableStateOf(false) }

    fun processImage(bitmap: Bitmap) {
        selectedImage = bitmap
        isAnalyzing = true
        try {
            val result = classifier.classify(bitmap)
            diagnosisResult = result
            val timeString = SimpleDateFormat("dd MMM, hh:mm a", Locale.getDefault()).format(Date())
            historyList.add(
                0,
                ScanHistoryItem(
                    timestamp = timeString,
                    className = result.className,
                    confidence = result.confidence,
                    latencyMs = result.latencyMs,
                    bitmap = bitmap
                )
            )
        } catch (e: Exception) {
            Toast.makeText(context, "Inference error: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
        } finally {
            isAnalyzing = false
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            val logoBmp = remember { loadBitmapFromAsset(context, "app_logo.png") }
                            if (logoBmp != null) {
                                Image(
                                    bitmap = logoBmp.asImageBitmap(),
                                    contentDescription = "CottonCare Logo",
                                    modifier = Modifier
                                        .size(38.dp)
                                        .clip(RoundedCornerShape(10.dp)),
                                    contentScale = ContentScale.Fit
                                )
                            } else {
                                Surface(
                                    shape = RoundedCornerShape(10.dp),
                                    color = Color(0xFF10B981).copy(alpha = 0.2f),
                                    border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.5f)),
                                    modifier = Modifier.size(38.dp)
                                ) {
                                    Box(contentAlignment = Alignment.Center) {
                                        Text("🌿", fontSize = 20.sp)
                                    }
                                }
                            }
                            Spacer(modifier = Modifier.width(10.dp))
                            Column {
                                Text(
                                    "CottonCare",
                                    fontWeight = FontWeight.ExtraBold,
                                    fontSize = 18.sp,
                                    color = Color.White
                                )
                                Text(
                                    "AI-Powered Leaf Disease Detection",
                                    fontSize = 10.sp,
                                    color = Color(0xFF34D399),
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        // Language Toggle Pills
                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(20.dp))
                                .background(Color(0xFF1E293B))
                                .border(1.dp, Color(0xFF334155), RoundedCornerShape(20.dp))
                                .padding(2.dp)
                        ) {
                            listOf("en" to "EN", "hi" to "हिंदी", "or" to "ଓଡ଼ିଆ").forEach { (code, label) ->
                                val isSelected = selectedLanguage == code
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(16.dp))
                                        .background(
                                            if (isSelected) Brush.horizontalGradient(
                                                listOf(Color(0xFF10B981), Color(0xFF059669))
                                            ) else Brush.horizontalGradient(listOf(Color.Transparent, Color.Transparent))
                                        )
                                        .clickable { selectedLanguage = code }
                                        .padding(horizontal = 8.dp, vertical = 4.dp)
                                ) {
                                    Text(
                                        text = label,
                                        fontSize = 11.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color.White else Color.LightGray
                                    )
                                }
                            }
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF0B131E)
                )
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = Color(0xFF0B131E),
                tonalElevation = 8.dp
            ) {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Icon(Icons.Default.CenterFocusStrong, contentDescription = "Scanner") },
                    label = { Text("Scanner", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) },
                    colors = NavigationBarItemDefaults.colors(
                        selectedIconColor = Color(0xFF10B981),
                        selectedTextColor = Color(0xFF10B981),
                        indicatorColor = Color(0xFF10B981).copy(alpha = 0.15f),
                        unselectedIconColor = Color(0xFF64748B),
                        unselectedTextColor = Color(0xFF64748B)
                    )
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Icon(Icons.Default.PhotoLibrary, contentDescription = "Specimens") },
                    label = { Text("Specimens", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) },
                    colors = NavigationBarItemDefaults.colors(
                        selectedIconColor = Color(0xFF10B981),
                        selectedTextColor = Color(0xFF10B981),
                        indicatorColor = Color(0xFF10B981).copy(alpha = 0.15f),
                        unselectedIconColor = Color(0xFF64748B),
                        unselectedTextColor = Color(0xFF64748B)
                    )
                )
                NavigationBarItem(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    icon = { Icon(Icons.Default.History, contentDescription = "Records") },
                    label = { Text("Records (${historyList.size})", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) },
                    colors = NavigationBarItemDefaults.colors(
                        selectedIconColor = Color(0xFF10B981),
                        selectedTextColor = Color(0xFF10B981),
                        indicatorColor = Color(0xFF10B981).copy(alpha = 0.15f),
                        unselectedIconColor = Color(0xFF64748B),
                        unselectedTextColor = Color(0xFF64748B)
                    )
                )
                NavigationBarItem(
                    selected = selectedTab == 3,
                    onClick = { selectedTab = 3 },
                    icon = { Icon(Icons.Default.Analytics, contentDescription = "Research") },
                    label = { Text("Research", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) },
                    colors = NavigationBarItemDefaults.colors(
                        selectedIconColor = Color(0xFF10B981),
                        selectedTextColor = Color(0xFF10B981),
                        indicatorColor = Color(0xFF10B981).copy(alpha = 0.15f),
                        unselectedIconColor = Color(0xFF64748B),
                        unselectedTextColor = Color(0xFF64748B)
                    )
                )
            }
        }
    ) { innerPadding ->
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            color = Color(0xFF070D14)
        ) {
            when (selectedTab) {
                0 -> ScanScreen(
                    selectedImage = selectedImage,
                    diagnosisResult = diagnosisResult,
                    isAnalyzing = isAnalyzing,
                    selectedLanguage = selectedLanguage,
                    onProcessImage = { bitmap -> processImage(bitmap) },
                    onClearImage = {
                        selectedImage = null
                        diagnosisResult = null
                    }
                )
                1 -> SamplesGalleryScreen(
                    onSelectSample = { bitmap ->
                        processImage(bitmap)
                        selectedTab = 0
                    }
                )
                2 -> HistoryScreen(
                    historyList = historyList,
                    onClear = { historyList.clear() }
                )
                3 -> BenchmarkScreen()
            }
        }
    }
}

// ==================== TAB 0: SCAN SCREEN ====================
@Composable
fun ScanScreen(
    selectedImage: Bitmap?,
    diagnosisResult: CottonNetClassifier.Recognition?,
    isAnalyzing: Boolean,
    selectedLanguage: String,
    onProcessImage: (Bitmap) -> Unit,
    onClearImage: () -> Unit
) {
    val context = LocalContext.current
    var isLiveCameraActive by remember { mutableStateOf(false) }

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            isLiveCameraActive = false
            onProcessImage(bitmap)
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            takePictureLauncher.launch(null)
        } else {
            Toast.makeText(context, "Camera permission needed to scan leaves.", Toast.LENGTH_SHORT).show()
        }
    }

    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            isLiveCameraActive = false
            val bmp = loadBitmapFromUri(context, uri)
            if (bmp != null) {
                onProcessImage(bmp)
            } else {
                Toast.makeText(context, "Unable to load image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val quickSampleList = listOf(
        "leaf_2.jpg" to "Bacterial Blight",
        "leaf_3.jpg" to "Curl Virus",
        "leaf_4.jpg" to "Healthy Leaf",
        "leaf_5.jpg" to "Herbicide Damage",
        "leaf_6.jpg" to "Leaf Hopper",
        "leaf_7.jpg" to "Leaf Redding",
        "leaf_8.jpg" to "Variegation"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        // ==================== VIEWFINDER CARD ====================
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(310.dp),
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
            border = BorderStroke(1.5.dp, Brush.verticalGradient(
                listOf(Color(0xFF10B981).copy(alpha = 0.6f), Color(0xFF047857).copy(alpha = 0.2f))
            ))
        ) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (isLiveCameraActive) {
                    CameraXLivePreview(
                        onImageCaptured = { bmp ->
                            isLiveCameraActive = false
                            onProcessImage(bmp)
                        }
                    )
                } else if (selectedImage != null) {
                    Image(
                        bitmap = selectedImage.asImageBitmap(),
                        contentDescription = "Specimen",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )

                    // Retake / Clear Pill
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(12.dp),
                        contentAlignment = Alignment.TopEnd
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = Color.Black.copy(alpha = 0.7f),
                            border = BorderStroke(1.dp, Color.White.copy(alpha = 0.2f)),
                            modifier = Modifier.clickable { onClearImage() }
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(Icons.Default.Refresh, contentDescription = "Retake", tint = Color.White, modifier = Modifier.size(14.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Retake", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                } else {
                    // Modern Viewfinder Framing Graphics
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(20.dp)
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = Color(0xFF10B981).copy(alpha = 0.12f),
                            border = BorderStroke(1.5.dp, Color(0xFF10B981).copy(alpha = 0.4f)),
                            modifier = Modifier.size(76.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    Icons.Default.CenterFocusStrong,
                                    contentDescription = "Target",
                                    tint = Color(0xFF34D399),
                                    modifier = Modifier.size(40.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(14.dp))
                        Text(
                            "Position Cotton Leaf in Frame",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            "Edge-AI extracts multi-spectral pathology\nwith 100% offline inference",
                            color = Color(0xFF94A3B8),
                            fontSize = 12.sp,
                            textAlign = TextAlign.Center,
                            lineHeight = 16.sp
                        )
                    }
                }

                if (isAnalyzing) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.75f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(
                                color = Color(0xFF10B981),
                                strokeWidth = 3.dp,
                                modifier = Modifier.size(36.dp)
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                "Analyzing leaf pathology...",
                                color = Color.White,
                                fontSize = 13.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // ==================== PRIMARY SCAN CONTROLS ====================
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Button(
                onClick = {
                    val hasPerm = ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED
                    if (hasPerm) {
                        takePictureLauncher.launch(null)
                    } else {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                },
                modifier = Modifier
                    .weight(1f)
                    .height(52.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF10B981)
                ),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp)
            ) {
                Icon(Icons.Default.CameraAlt, contentDescription = "Camera", tint = Color(0xFF042F2E))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Take Photo", color = Color(0xFF042F2E), fontWeight = FontWeight.ExtraBold, fontSize = 14.sp)
            }

            Button(
                onClick = { pickImageLauncher.launch("image/*") },
                modifier = Modifier
                    .weight(1f)
                    .height(52.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B)),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Icon(Icons.Default.PhotoLibrary, contentDescription = "Gallery", tint = Color.White)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Pick Gallery", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // ==================== QUICK SPECIMEN SELECTOR CHIPS ====================
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("⚡ QUICK TEST:", fontSize = 10.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFF64748B), letterSpacing = 1.sp)
        }
        Spacer(modifier = Modifier.height(6.dp))

        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            items(quickSampleList) { (file, label) ->
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = Color(0xFF131D28),
                    border = BorderStroke(1.dp, Color(0xFF1E293B)),
                    modifier = Modifier.clickable {
                        val bmp = loadBitmapFromAsset(context, "samples/$file")
                        if (bmp != null) onProcessImage(bmp)
                    }
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("🍃", fontSize = 11.sp)
                        Spacer(modifier = Modifier.width(5.dp))
                        Text(label, color = Color(0xFFE2E8F0), fontSize = 11.sp, fontWeight = FontWeight.Medium)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // ==================== DIAGNOSIS RESULT CARD ====================
        if (diagnosisResult != null) {
            val advisoryMap = DISEASE_ADVISORY[diagnosisResult.className]
            val advisory = advisoryMap?.get(selectedLanguage) ?: advisoryMap?.get("en")

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
                border = BorderStroke(1.5.dp, advisory?.severityColor?.copy(alpha = 0.5f) ?: Color(0xFF10B981).copy(alpha = 0.5f))
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    // Header Row: Condition & Confidence
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                            Text(advisory?.icon ?: "🍃", fontSize = 32.sp)
                            Spacer(modifier = Modifier.width(10.dp))
                            Column {
                                Text(
                                    "PRIMARY DIAGNOSIS",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.ExtraBold,
                                    color = advisory?.severityColor ?: Color(0xFF10B981),
                                    letterSpacing = 1.sp
                                )
                                Text(
                                    advisory?.englishName ?: diagnosisResult.className,
                                    fontSize = 17.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color.White
                                )
                            }
                        }

                        Surface(
                            shape = CircleShape,
                            color = (advisory?.severityColor ?: Color(0xFF10B981)).copy(alpha = 0.15f),
                            border = BorderStroke(1.5.dp, advisory?.severityColor ?: Color(0xFF10B981))
                        ) {
                            Text(
                                "${(diagnosisResult.confidence * 100).toInt()}%",
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                color = advisory?.severityColor ?: Color(0xFF10B981),
                                fontWeight = FontWeight.ExtraBold,
                                fontSize = 16.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    // Badges Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = Color(0xFF1E293B)
                        ) {
                            Text(
                                "⚡ Latency: ${diagnosisResult.latencyMs} ms",
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                                color = Color(0xFFCBD5E1),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }

                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = (advisory?.severityColor ?: Color(0xFF10B981)).copy(alpha = 0.15f)
                        ) {
                            Text(
                                advisory?.severity ?: "Confirmed",
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                                color = advisory?.severityColor ?: Color(0xFF10B981),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    HorizontalDivider(modifier = Modifier.padding(vertical = 14.dp), color = Color(0xFF1E293B))

                    // ==================== PRESCRIPTION SECTION ====================
                    if (advisory != null) {
                        Text("🔍 SYMPTOMS & PATHOLOGY", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF94A3B8))
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(advisory.symptoms, fontSize = 13.sp, color = Color(0xFFF1F5F9), lineHeight = 18.sp)

                        Spacer(modifier = Modifier.height(12.dp))

                        Text("💊 CHEMICAL TREATMENT & DOSAGE", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF38BDF8))
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(advisory.chemicalControl, fontSize = 13.sp, color = Color(0xFFF1F5F9), lineHeight = 18.sp)

                        Spacer(modifier = Modifier.height(12.dp))

                        Text("🌱 BIOLOGICAL & ORGANIC DEFENSE", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF4ADE80))
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(advisory.organicControl, fontSize = 13.sp, color = Color(0xFFF1F5F9), lineHeight = 18.sp)

                        HorizontalDivider(modifier = Modifier.padding(vertical = 14.dp), color = Color(0xFF1E293B))
                    }

                    // ==================== PROBABILITY DISTRIBUTION ====================
                    Text("📊 SOFTMAX PROBABILITY SPECTRUM", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF94A3B8))
                    Spacer(modifier = Modifier.height(8.dp))

                    CottonNetClassifier.CLASS_NAMES.forEachIndexed { idx, name ->
                        val prob = if (idx < diagnosisResult.allProbabilities.size) diagnosisResult.allProbabilities[idx] else 0f
                        val isTop = idx == diagnosisResult.classIndex
                        Column(modifier = Modifier.padding(vertical = 3.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    name,
                                    fontSize = 12.sp,
                                    color = if (isTop) Color(0xFF34D399) else Color(0xFF94A3B8),
                                    fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal
                                )
                                Text(
                                    "${(prob * 100).toInt()}%",
                                    fontSize = 12.sp,
                                    color = if (isTop) Color(0xFF34D399) else Color(0xFF64748B),
                                    fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal
                                )
                            }
                            Spacer(modifier = Modifier.height(3.dp))
                            LinearProgressIndicator(
                                progress = { prob },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                                color = if (isTop) Color(0xFF10B981) else Color(0xFF334155),
                                trackColor = Color(0xFF1E293B)
                            )
                        }
                    }
                }
            }
        }
    }
}

// ==================== CAMERAX LIVE PREVIEW COMPOSABLE ====================
@Composable
fun CameraXLivePreview(
    onImageCaptured: (Bitmap) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var imageCapture: ImageCapture? by remember { mutableStateOf(null) }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    try {
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        val capture = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .build()
                        imageCapture = capture

                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            capture
                        )
                    } catch (e: Exception) {
                        Log.e("CottonGuard", "Camera binding failed", e)
                    }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        Button(
            onClick = {
                val capture = imageCapture ?: return@Button
                val executor = Executors.newSingleThreadExecutor()
                capture.takePicture(
                    executor,
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(imageProxy: ImageProxy) {
                            val bitmap = imageProxy.toBitmap()
                            imageProxy.close()
                            ContextCompat.getMainExecutor(context).execute {
                                onImageCaptured(bitmap)
                            }
                        }

                        override fun onError(exception: ImageCaptureException) {
                            Log.e("CottonGuard", "Capture error", exception)
                        }
                    }
                )
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981))
        ) {
            Icon(Icons.Default.CameraAlt, contentDescription = "Capture", tint = Color.Black)
            Spacer(modifier = Modifier.width(6.dp))
            Text("Capture Leaf", color = Color.Black, fontWeight = FontWeight.Bold)
        }
    }
}

// ==================== TAB 1: SAMPLES GALLERY SCREEN ====================
@Composable
fun SamplesGalleryScreen(
    onSelectSample: (Bitmap) -> Unit
) {
    val context = LocalContext.current

    val sampleFiles = listOf(
        "leaf_2.jpg" to "Bacterial Blight",
        "leaf_3.jpg" to "Curl Virus",
        "leaf_4.jpg" to "Healthy Leaf",
        "leaf_5.jpg" to "Herbicide Growth Damage",
        "leaf_6.jpg" to "Leaf Hopper Jassids",
        "leaf_7.jpg" to "Leaf Redding",
        "leaf_8.jpg" to "Leaf Variegation",
        "leaf_9.jpg" to "Bacterial Blight Specimen 2",
        "leaf_10.jpg" to "Curl Virus Specimen 2",
        "leaf_11.jpg" to "Healthy Leaf Specimen 2",
        "leaf_13.jpg" to "Herbicide Damage Specimen 2",
        "leaf_36.jpg" to "Hopper Burn Specimen 2"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("🌿 Cotton Pathology Test Gallery", fontWeight = FontWeight.ExtraBold, fontSize = 18.sp, color = Color.White)
        Text("Select any authentic leaf specimen to run instant CottonNet inference.", color = Color(0xFF94A3B8), fontSize = 12.sp)
        Spacer(modifier = Modifier.height(14.dp))

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier.fillMaxSize()
        ) {
            items(sampleFiles) { (filename, label) ->
                val bmp = remember(filename) { loadBitmapFromAsset(context, "samples/$filename") }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            if (bmp != null) onSelectSample(bmp)
                            else Toast.makeText(context, "Sample not found", Toast.LENGTH_SHORT).show()
                        },
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
                    border = BorderStroke(1.dp, Color(0xFF1E293B))
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        if (bmp != null) {
                            Image(
                                bitmap = bmp.asImageBitmap(),
                                contentDescription = label,
                                modifier = Modifier
                                    .size(60.dp)
                                    .clip(RoundedCornerShape(10.dp)),
                                contentScale = ContentScale.Crop
                            )
                        } else {
                            Box(
                                modifier = Modifier
                                    .size(60.dp)
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(Color(0xFF1E293B)),
                                contentAlignment = Alignment.Center
                            ) {
                                Text("🌿", fontSize = 24.sp)
                            }
                        }

                        Spacer(modifier = Modifier.width(14.dp))

                        Column(modifier = Modifier.weight(1f)) {
                            Text(label, fontWeight = FontWeight.Bold, color = Color.White, fontSize = 15.sp)
                            Text("Specimen: $filename • Tap to Test", color = Color(0xFF34D399), fontSize = 12.sp)
                        }

                        Icon(Icons.Default.ChevronRight, contentDescription = "Test", tint = Color(0xFF64748B))
                    }
                }
            }
        }
    }
}

// ==================== TAB 2: HISTORY SCREEN ====================
@Composable
fun HistoryScreen(
    historyList: List<ScanHistoryItem>,
    onClear: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("📋 Diagnostic Scan Log", fontWeight = FontWeight.ExtraBold, fontSize = 18.sp, color = Color.White)
                Text("Saved locally on your mobile device", color = Color(0xFF94A3B8), fontSize = 12.sp)
            }
            if (historyList.isNotEmpty()) {
                TextButton(onClick = onClear) {
                    Text("Clear All", color = Color(0xFFEF4444), fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        if (historyList.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Default.History, contentDescription = "No History", tint = Color(0xFF334155), modifier = Modifier.size(64.dp))
                    Spacer(modifier = Modifier.height(12.dp))
                    Text("No scans saved yet.", color = Color(0xFF94A3B8), fontSize = 15.sp, fontWeight = FontWeight.Medium)
                    Text("Capture or upload a leaf photo to begin.", color = Color(0xFF64748B), fontSize = 12.sp)
                }
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.fillMaxSize()
            ) {
                items(historyList) { item ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(14.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
                        border = BorderStroke(1.dp, Color(0xFF1E293B))
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            if (item.bitmap != null) {
                                Image(
                                    bitmap = item.bitmap.asImageBitmap(),
                                    contentDescription = item.className,
                                    modifier = Modifier
                                        .size(54.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    contentScale = ContentScale.Crop
                                )
                            } else {
                                Surface(
                                    shape = RoundedCornerShape(8.dp),
                                    color = Color(0xFF1E293B),
                                    modifier = Modifier.size(54.dp)
                                ) {
                                    Box(contentAlignment = Alignment.Center) {
                                        Text("🌱", fontSize = 22.sp)
                                    }
                                }
                            }

                            Spacer(modifier = Modifier.width(12.dp))

                            Column(modifier = Modifier.weight(1f)) {
                                Text(item.className, fontWeight = FontWeight.Bold, color = Color.White, fontSize = 15.sp)
                                Text("${item.timestamp} • ${item.latencyMs}ms", color = Color(0xFF64748B), fontSize = 11.sp)
                            }

                            Surface(
                                shape = CircleShape,
                                color = Color(0xFF10B981).copy(alpha = 0.15f),
                                border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.5f))
                            ) {
                                Text(
                                    "${(item.confidence * 100).toInt()}%",
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                                    color = Color(0xFF34D399),
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 12.sp
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// ==================== TAB 3: BENCHMARK SCREEN ====================
@Composable
fun BenchmarkScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Text("🔬 CottonNet Clinical Benchmark", fontWeight = FontWeight.ExtraBold, fontSize = 18.sp, color = Color.White)
        Text("SAR-CLD-2024 Evaluation Dataset Results", color = Color(0xFF34D399), fontSize = 12.sp)

        Spacer(modifier = Modifier.height(14.dp))

        // Summary Metric Cards
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            MetricCard(title = "Test F1 Score", value = "98.60%", subtitle = "SAR-CLD-2024", modifier = Modifier.weight(1f))
            MetricCard(title = "Parameters", value = "4.87 M", subtitle = "Lightweight", modifier = Modifier.weight(1f))
            MetricCard(title = "Edge INT8", value = "5.18 MB", subtitle = "Quantized", modifier = Modifier.weight(1f))
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
            border = BorderStroke(1.dp, Color(0xFF1E293B))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model Architecture Benchmark", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 15.sp)
                Spacer(modifier = Modifier.height(12.dp))

                ComparisonRow(model = "CottonNet (Ours)", params = "4.87M", f1 = "98.60%", isHighlighted = true)
                ComparisonRow(model = "DenseNet-121", params = "7.98M", f1 = "97.40%", isHighlighted = false)
                ComparisonRow(model = "MobileNetV3-Large", params = "5.48M", f1 = "96.80%", isHighlighted = false)
                ComparisonRow(model = "ResNet-50", params = "25.6M", f1 = "97.10%", isHighlighted = false)
                ComparisonRow(model = "EfficientNet-B0", params = "5.29M", f1 = "96.50%", isHighlighted = false)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
            border = BorderStroke(1.dp, Color(0xFF1E293B))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("🛡️ Field Robustness & OOD Integrity", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 15.sp)
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Gaussian / Optical Defocus Blur: 94.2% Retention", color = Color(0xFFCBD5E1), fontSize = 13.sp)
                Text("• Fast Motion / Hand Jitter Blur: 91.8% Retention", color = Color(0xFFCBD5E1), fontSize = 13.sp)
                Text("• Low Light & Deep Shadows: 93.5% Retention", color = Color(0xFFCBD5E1), fontSize = 13.sp)
                Text("• Out-of-Distribution Detection: 99.1% AUROC", color = Color(0xFFCBD5E1), fontSize = 13.sp)
            }
        }
    }
}

@Composable
fun MetricCard(title: String, value: String, subtitle: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF101924)),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, fontSize = 10.sp, color = Color(0xFF94A3B8), fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(4.dp))
            Text(value, fontSize = 16.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFF34D399))
            Spacer(modifier = Modifier.height(2.dp))
            Text(subtitle, fontSize = 9.sp, color = Color(0xFF64748B))
        }
    }
}

@Composable
fun ComparisonRow(model: String, params: String, f1: String, isHighlighted: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                if (isHighlighted) Color(0xFF10B981).copy(alpha = 0.12f) else Color.Transparent,
                RoundedCornerShape(8.dp)
            )
            .padding(vertical = 8.dp, horizontal = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            model,
            color = if (isHighlighted) Color(0xFF34D399) else Color.White,
            fontWeight = if (isHighlighted) FontWeight.Bold else FontWeight.Normal,
            fontSize = 13.sp
        )
        Row {
            Text(params, color = Color(0xFF94A3B8), fontSize = 12.sp)
            Spacer(modifier = Modifier.width(16.dp))
            Text(
                f1,
                color = if (isHighlighted) Color(0xFF34D399) else Color(0xFFCBD5E1),
                fontWeight = if (isHighlighted) FontWeight.Bold else FontWeight.Normal,
                fontSize = 13.sp
            )
        }
    }
}

// ==================== THEME ====================
@Composable
fun CottonGuardTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF10B981),
            secondary = Color(0xFF059669),
            background = Color(0xFF070D14),
            surface = Color(0xFF101924)
        ),
        content = content
    )
}
