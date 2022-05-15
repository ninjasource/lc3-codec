#![allow(clippy::all)]

// Spectral Noise Shaping Quantization Tables
use crate::common::complex::Scaler;

// off-line trained stochastic low-frequency codebook
#[rustfmt::skip]
pub const LFCB: [[Scaler; 8]; 32] = [
    [
        2.262833655926780e+00, 8.133112690613385e-01, -5.301934948714359e-01, -1.356648359034418e+00,
        -1.599521765631959e+00, -1.440987684300950e+00, -1.143816483058210e+00, -7.552037679090641e-01,
    ],
    [
        2.945164791913764e+00, 2.411433179566788e+00, 9.604551064007274e-01, -4.432264880769172e-01,
        -1.229136124255896e+00, -1.555900391181699e+00, -1.496886559523759e+00, -1.116899865014692e+00,
    ],
    [
        -2.186107070099790e+00, -1.971521356752276e+00, -1.787186196810059e+00, -1.918658956855768e+00,
        -1.793991218365963e+00, -1.357384042572884e+00, -7.054442793538694e-01, -4.781729447777114e-02,
    ],
    [
        6.936882365289195e-01, 9.556098571582197e-01, 5.752307870387333e-01, -1.146034194628886e-01,
        -6.460506374360290e-01, -9.523513704496247e-01, -1.074052472261504e+00, -7.580877070949045e-01,
    ],
    [
        -1.297521323152956e+00, -7.403690571778526e-01, -3.453724836421064e-01, -3.132856962479401e-01,
        -4.029772428244766e-01, -3.720208534652272e-01, -7.834141773237381e-02, 9.704413039922949e-02,
    ],
    [
        9.146520378306716e-01, 1.742930434352573e+00, 1.909066268599861e+00, 1.544084838426651e+00,
        1.093449607614550e+00, 6.474795495182776e-01, 3.617907524496421e-02, -2.970928071788889e-01,
    ],
    [
        -2.514288125789621e+00, -2.891752713843728e+00, -2.004506667594338e+00, -7.509122739031269e-01,
        4.412021049046914e-01, 1.201909876010087e+00, 1.327428572572904e+00, 1.220490811409839e+00,
    ],
    [
        -9.221884048123851e-01, 6.324951414405520e-01, 1.087364312546411e+00, 6.086286245358197e-01,
        1.311745675473482e-01, -2.961491577437521e-01, -2.070135165256287e-01, 1.349249166420795e-01,
    ],
    [
        7.903222883692664e-01, 6.284012618761988e-01, 3.931179235404499e-01, 4.800077108669007e-01,
        4.478151380501427e-01, 2.097342145522343e-01, 6.566919964280205e-03, -8.612423420618573e-02,
    ],
    [
        1.447755801787238e+00, 2.723999516749523e+00, 2.310832687375278e+00, 9.350512695665294e-01,
        -2.747439113836877e-01, -9.020776968286019e-01, -9.406815119454044e-01, -6.336970389743102e-01,
    ],
    [
        7.933545264174744e-01, 1.439311855234535e-02, -5.678348447296789e-01, -6.547604679167449e-01,
        -4.794589984757430e-01, -1.738946619028885e-01, 6.801627055154381e-02, 2.951259483697938e-01,
    ],
    [
        2.724253473850336e+00, 2.959475724048243e+00, 1.849535592684608e+00, 5.632849223223643e-01,
        1.399170881250724e-01, 3.596410933662221e-01, 6.894613547745887e-01, 6.397901768331046e-01,
    ],
    [
        -5.308301983754000e-01, -2.126906828121638e-01, 5.766136283770966e-03, 4.248714843837454e-01,
        4.731289521586675e-01, 8.588941993212806e-01, 1.191111608544352e+00, 9.961896696383581e-01,
    ],
    [
        1.687284108450062e+00, 2.436145092376558e+00, 2.330194290782250e+00, 1.779837778350905e+00,
        1.444112953900818e+00, 1.519951770097301e+00, 1.471993937504249e+00, 9.776824738917613e-01,
    ],
    [
        -2.951832728018580e+00, -1.593934967733454e+00, -1.099187728780224e-01, 3.886090729192574e-01,
        5.129326495175837e-01, 6.281125970634966e-01, 8.226217964306339e-01, 8.758914246550805e-01,
    ],
    [
        1.018783427856281e-01, 5.898573242289165e-01, 6.190476467934656e-01, 1.267313138517963e+00,
        2.419610477698038e+00, 2.251742525721865e+00, 5.265370309912005e-01, -3.965915132279989e-01,
    ],
    [
        2.682545754984259e+00, 1.327380108994199e+00, 1.301852738040482e-01, -3.385330885113471e-01,
        -3.682192358996665e-01, -1.916899467159607e-01, -1.547823771539079e-01, -2.342071777743923e-01,
    ],
    [
        4.826979236804030e+00, 3.119478044924880e+00, 1.395136713851784e+00, 2.502953159187215e-01,
        -3.936138393797931e-01, -6.434581730547007e-01, -6.425707368569433e-01, -7.231932234440720e-01,
    ],
    [
        8.784199364703349e-02, -5.695868402385010e-01, -1.145060156688110e+00, -1.669684881725975e+00,
        -1.845344176036817e+00, -1.564680273288019e+00, -1.117467590764198e+00, -5.339816633667862e-01,
    ],
    [
        1.391023082043259e+00, 1.981464791994655e+00, 1.112657963887701e+00, -2.201075094207434e-01,
        -7.749656115523655e-01, -5.940638741491173e-01, 1.369376806289231e-01, 8.182428912643381e-01,
    ],
    [
        3.845858938891820e-01, -1.605887855365100e-01, -5.393668095577095e-01, -5.293090787898571e-01,
        1.904335474379324e-01, 2.560629181065215e+00, 2.818963982452484e+00, 6.566708756961611e-01,
    ],
    [
        1.932273994417191e+00, 3.010301804120569e+00, 3.065438938262036e+00, 2.501101608700079e+00,
        1.930895929789344e+00, 5.721538109618367e-01, -8.117417940810907e-01, -1.176418108619025e+00,
    ],
    [
        1.750804628998837e-01, -7.505228322489846e-01, -1.039438933422309e+00, -1.135775089376484e+00,
        -1.041979038374938e+00, -1.520600989933816e-02, 2.070483917167066e+00, 3.429489180816891e+00,
    ],
    [
        -1.188170202505555e+00, 3.667928736626364e-01, 1.309578304090959e+00, 1.683306872804914e+00,
        1.251009242251268e+00, 9.423757516286146e-01, 8.262504833741330e-01, 4.399527411209563e-01,
    ],
    [
        2.533222033270612e+00, 2.112746426959081e+00, 1.262884115020644e+00, 7.615135124304274e-01,
        5.221179379761699e-01, 1.186800697571213e-01, -4.523468275073703e-01, -7.003524261611032e-01,
    ],
    [
        3.998898374856063e+00, 4.079017514519560e+00, 2.822856611024964e+00, 1.726072128495800e+00,
        6.471443773486192e-01, -3.311485212172380e-01, -8.840425708487493e-01, -1.126973406454781e+00,
    ],
    [
        5.079025931863813e-01, 1.588384497895265e+00, 1.728990238692094e+00, 1.006922302417256e+00,
        3.771212318163816e-01, 4.763707668994976e-01, 1.087547403721699e+00, 1.087562660992209e+00,
    ],
    [
        3.168568251075689e+00, 3.258534581594065e+00, 2.422305913285988e+00, 1.794460776432612e+00,
        1.521779106530886e+00, 1.171967065376021e+00, 4.893945969806952e-01, -6.227957157187685e-02,
    ],
    [
        1.894147667317636e+00, 1.251086946092320e+00, 5.904512107206275e-01, 6.083585832937136e-01,
        8.781710100110816e-01, 1.119125109509496e+00, 1.018576615503421e+00, 6.204538910117241e-01,
    ],
    [
        9.488806045171881e-01, 2.132394392499823e+00, 2.723453503442780e+00, 2.769860768665877e+00,
        2.542869732549456e+00, 2.020462638250194e+00, 8.300458594009102e-01, -2.755691738882634e-02,
    ],
    [
        -1.880267570456275e+00, -1.264310727587049e+00, 3.114249769686986e-01, 1.836702103064300e+00,
        2.256341918398738e+00, 2.048189984634735e+00, 2.195268374585677e+00, 2.026596138366193e+00,
    ],
    [
        2.463757462771289e-01, 9.556217733930993e-01, 1.520467767417663e+00, 1.976474004194571e+00,
        1.940438671774617e+00, 2.233758472826862e+00, 1.988359777584072e+00, 1.272326725547010e+00,
    ],
];

// off-line trained stochastic high-frequency codebook
#[rustfmt::skip]
pub const HFCB: [[Scaler; 8]; 32] = [
    [
        2.320284191244650e-01, -1.008902706044547e+00, -2.142235027894714e+00, -2.375338135706641e+00,
        -2.230419330496551e+00, -2.175958812236960e+00, -2.290659135409999e+00, -2.532863979798455e+00,
    ],
    [
        -1.295039366736175e+00, -1.799299653843385e+00, -1.887031475315188e+00, -1.809916596873323e+00,
        -1.763400384792061e+00, -1.834184284679500e+00, -1.804809806874051e+00, -1.736795453174010e+00,
    ],
    [
        1.392857160458027e-01, -2.581851261717519e-01, -6.508045726701103e-01, -1.068157317819692e+00,
        -1.619287415243023e+00, -2.187625664417564e+00, -2.637575869390537e+00, -2.978977495750963e+00,
    ],
    [
        -3.165131021857248e-01, -4.777476572098050e-01, -5.511620758797545e-01, -4.847882833811970e-01,
        -2.383883944558142e-01, -1.430245072855038e-01, 6.831866736490735e-02, 8.830617172880660e-02,
    ],
    [
        8.795184052264962e-01, 2.983400960071886e-01, -9.153863964057101e-01, -2.206459747397620e+00,
        -2.741421809599509e+00, -2.861390742768913e+00, -2.888415971052714e+00, -2.951826082625207e+00,
    ],
    [
        -2.967019224553751e-01, -9.750049191745525e-01, -1.358575002469926e+00, -9.837211058374442e-01,
        -6.529569391008090e-01, -9.899869929218105e-01, -1.614672245988999e+00, -2.407123023851163e+00,
    ],
    [
        3.409811004696971e-01, 2.688997889460545e-01, 5.633356848280326e-02, 4.991140468266853e-02,
        -9.541307274143691e-02, -7.601661460838854e-01, -2.327581201770068e+00, -3.771554853856562e+00,
    ],
    [
        -1.412297590775968e+00, -1.485221193498518e+00, -1.186035798347001e+00, -6.250016344413516e-01,
        1.539024974683036e-01, 5.763864978107553e-01, 7.950926037988714e-01, 5.965646321449126e-01,
    ],
    [
        -2.288395118273794e-01, -3.337190697846616e-01, -8.093213593246560e-01, -1.635878769237973e+00,
        -1.884863973309819e+00, -1.644966913163562e+00, -1.405157780466116e+00, -1.466664713261457e+00,
    ],
    [
        -1.071486285444486e+00, -1.417670154562606e+00, -1.548917622654407e+00, -1.452960624755303e+00,
        -1.031829700622701e+00, -6.906426402725842e-01, -4.288438045321706e-01, -4.949602154088736e-01,
    ],
    [
        -5.909885111880511e-01, -7.117377585376282e-02, 3.457195229473127e-01, 3.005494609962507e-01,
        -1.118652182958568e+00, -2.440891511480490e+00, -2.228547324507349e+00, -1.895092282108533e+00,
    ],
    [
        -8.484340988361639e-01, -5.832268107088888e-01, 9.004236881428734e-02, 8.450250075568864e-01,
        1.065723845017161e+00, 7.375829993777555e-01, 2.565904524599121e-01, -4.919633597623784e-01,
    ],
    [
        1.140691455623824e+00, 9.640168923982929e-01, 3.814612059847975e-01, -4.828493406089983e-01,
        -1.816327212605887e+00, -2.802795127285548e+00, -3.233857248338638e+00, -3.459087144914729e+00,
    ],
    [
        -3.762832379674643e-01, 4.256754620961052e-02, 5.165476965923055e-01, 2.517168818646298e-01,
        -2.161799675243032e-01, -5.340740911245042e-01, -6.407860962621957e-01, -8.697450323741350e-01,
    ],
    [
        6.650041205984020e-01, 1.097907646907945e+00, 1.383426671120792e+00, 1.343273586282854e+00,
        8.229788368559223e-01, 2.158767985156789e-01, -4.049257530802925e-01, -1.070256058705229e+00,
    ],
    [
        -8.262659539826793e-01, -6.711812327666034e-01, -2.284955927794715e-01, 5.189808525519373e-01,
        1.367218963402784e+00, 2.180230382530922e+00, 2.535960927501071e+00, 2.201210988600361e+00,
    ],
    [
        1.410083268321729e+00, 7.544419078354684e-01, -1.305505849586310e+00, -1.871337113509707e+00,
        -1.240086851563054e+00, -1.267129248662737e+00, -2.036708130039070e+00, -2.896851622423807e+00,
    ],
    [
        3.613868175743476e-01, -2.199917054278258e-02, -5.793688336338242e-01, -8.794279609410701e-01,
        -8.506850234081188e-01, -7.793970501558157e-01, -7.321829272918255e-01, -8.883485148212548e-01,
    ],
    [
        4.374692393303287e-01, 3.054404196059607e-01, -7.387865664783739e-03, -4.956498547102520e-01,
        -8.066512711183929e-01, -1.224318919844005e+00, -1.701577700431810e+00, -2.244919137556108e+00,
    ],
    [
        6.481003189965029e-01, 6.822991336406795e-01, 2.532474643329756e-01, 7.358421437884688e-02,
        3.142167093890103e-01, 2.347298809236790e-01, 1.446001344798368e-01, -6.821201788801744e-02,
    ],
    [
        1.119198330913041e+00, 1.234655325360046e+00, 5.891702380853181e-01, -1.371924596531664e+00,
        -2.370957072415767e+00, -2.007797826823599e+00, -1.666885402243946e+00, -1.926318462584058e+00,
    ],
    [
        1.418474970871759e-01, -1.106600706331509e-01, -2.828245925436287e-01, -6.598134746141936e-03,
        2.859292796272158e-01, 4.604455299529710e-02, -6.025964155778858e-01, -2.265687286325748e+00,
    ],
    [
        5.040469553902519e-01, 8.269821629590972e-01, 1.119812362918282e+00, 1.179140443327336e+00,
        1.079874291972597e+00, 6.975362390675000e-01, -9.125488173710808e-01, -3.576847470627726e+00,
    ],
    [
        -5.010760504793567e-01, -3.256780060814170e-01, 2.807981949470768e-02, 2.620545547631326e-01,
        3.605908060857668e-01, 6.356237220536995e-01, 9.590124671781544e-01, 1.307451566886533e+00,
    ],
    [
        3.749709827096420e+00, 1.523426118470452e+00, -4.577156618978547e-01, -7.987110082431923e-01,
        -3.868193293091003e-01, -3.759010622312032e-01, -6.578368999305377e-01, -1.281639642436027e+00,
    ],
    [
        -1.152589909805491e+00, -1.108008859062412e+00, -5.626151165124718e-01, -2.205621237656746e-01,
        -3.498428803366437e-01, -7.534327702504950e-01, -9.885965933963837e-01, -1.287904717914711e+00,
    ],
    [
        1.028272464221398e+00, 1.097705193898282e+00, 7.686455457647760e-01, 2.060819777407656e-01,
        -3.428057350919982e-01, -7.549394046253397e-01, -1.041961776319998e+00, -1.503356529555287e+00,
    ],
    [
        1.288319717078174e-01, 6.894393952648783e-01, 1.123469050095749e+00, 1.309345231065936e+00,
        1.355119647139345e+00, 1.423113814707990e+00, 1.157064491909045e+00, 4.063194375168383e-01,
    ],
    [
        1.340330303347565e+00, 1.389968250677893e+00, 1.044679217088833e+00, 6.358227462443666e-01,
        -2.747337555184823e-01, -1.549233724306950e+00, -2.442397102780069e+00, -3.024576069445502e+00,
    ],
    [
        2.138431054193125e+00, 4.247112673031041e+00, 2.897341098304393e+00, 9.327306580268148e-01,
        -2.928222497298096e-01, -8.104042968531823e-01, -7.888680987564828e-01, -9.353531487613377e-01,
    ],
    [
        5.648304873553961e-01, 1.591849779587432e+00, 2.397716990151462e+00, 3.036973436007040e+00,
        2.664243503371508e+00, 1.393044850326060e+00, 4.038340235957454e-01, -6.562709713281135e-01,
    ],
    [
        -4.224605475860865e-01, 3.261496250498011e-01, 1.391713133422612e+00, 2.231466146364735e+00,
        2.611794421696881e+00, 2.665403401965702e+00, 2.401035541057067e+00, 1.759203796708810e+00,
    ],
];

pub const SNS_VQ_REG_ADJ_GAINS: [Scaler; 2] = [8915.0 / 4096.0, 12054.0 / 4096.0];
pub const SNS_VQ_REG_LF_ADJ_GAINS: [Scaler; 4] =
    [6245.0 / 4096.0, 15043.0 / 4096.0, 17861.0 / 4096.0, 21014.0 / 4096.0];
pub const SNS_VQ_NEAR_ADJ_GAINS: [Scaler; 4] = [7099.0 / 4096.0, 9132.0 / 4096.0, 11253.0 / 4096.0, 14808.0 / 4096.0];
pub const SNS_VQ_FAR_ADJ_GAINS: [Scaler; 8] = [
    4336.0 / 4096.0,
    5067.0 / 4096.0,
    5895.0 / 4096.0,
    8149.0 / 4096.0,
    10235.0 / 4096.0,
    12825.0 / 4096.0,
    16868.0 / 4096.0,
    19882.0 / 4096.0,
];

pub const SNS_GAIN_MSB_BITS: [usize; 4] = [1, 1, 2, 2];
pub const SNS_GAIN_LSB_BITS: [usize; 4] = [0, 1, 0, 1];

pub const MPVQ_OFFSETS: [[usize; 11]; 16] = [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    [0, 1, 5, 13, 25, 41, 61, 85, 113, 145, 181],
    [0, 1, 7, 25, 63, 129, 231, 377, 575, 833, 1159],
    [0, 1, 9, 41, 129, 321, 681, 1289, 2241, 3649, 5641],
    [0, 1, 11, 61, 231, 681, 1683, 3653, 7183, 13073, 22363],
    [0, 1, 13, 85, 377, 1289, 3653, 8989, 19825, 40081, 75517],
    [0, 1, 15, 113, 575, 2241, 7183, 19825, 48639, 108545, 224143],
    [0, 1, 17, 145, 833, 3649, 13073, 40081, 108545, 265729, 598417],
    [0, 1, 19, 181, 1159, 5641, 22363, 75517, 224143, 598417, 1462563],
    [0, 1, 21, 221, 1561, 8361, 36365, 134245, 433905, 1256465, 3317445],
    [0, 1, 23, 265, 2047, 11969, 56695, 227305, 795455, 2485825, 7059735],
    [0, 1, 25, 313, 2625, 16641, 85305, 369305, 1392065, 4673345, 14218905],
    [0, 1, 27, 365, 3303, 22569, 124515, 579125, 2340495, 8405905, 27298155],
    [0, 1, 29, 421, 4089, 29961, 177045, 880685, 3800305, 14546705, 50250765],
    [0, 1, 31, 481, 4991, 39041, 246047, 1303777, 5984767, 24331777, 89129247],
];

#[rustfmt::skip]
pub const D: [[Scaler; 16]; 16] = [
    // D is a rotation matrix
    // D consists of the base vectors of the DCT (orthogonalized DCT-II)
    // (the DCT base vector are stored in column-wise in this table)
    // first row results in the first coeff in fwd synthesis (dec+(enc))
    // first column results in the first coeff in the analysis(encoder)
    [
        2.500000000000000e-01, 3.518509343815957e-01, 3.467599613305369e-01, 3.383295002935882e-01,
        3.266407412190941e-01, 3.118062532466678e-01, 2.939689006048397e-01, 2.733004667504394e-01,
        2.500000000000001e-01, 2.242918965856591e-01, 1.964237395967756e-01, 1.666639146194367e-01,
        1.352990250365493e-01, 1.026311318805893e-01, 6.897484482073578e-02, 3.465429229977293e-02,
    ],
    [
        2.500000000000000e-01, 3.383295002935882e-01, 2.939689006048397e-01, 2.242918965856591e-01,
        1.352990250365493e-01, 3.465429229977286e-02, -6.897484482073579e-02, -1.666639146194366e-01,
        -2.500000000000001e-01, -3.118062532466678e-01, -3.467599613305369e-01, -3.518509343815956e-01,
        -3.266407412190941e-01, -2.733004667504394e-01, -1.964237395967756e-01, -1.026311318805893e-01,
    ],
    [
        2.500000000000000e-01, 3.118062532466678e-01, 1.964237395967756e-01, 3.465429229977286e-02,
        -1.352990250365493e-01, -2.733004667504394e-01, -3.467599613305369e-01, -3.383295002935882e-01,
        -2.500000000000001e-01, -1.026311318805894e-01, 6.897484482073574e-02, 2.242918965856590e-01,
        3.266407412190941e-01, 3.518509343815957e-01, 2.939689006048397e-01, 1.666639146194367e-01,
    ],
    [
        2.500000000000000e-01, 2.733004667504394e-01, 6.897484482073575e-02, -1.666639146194366e-01,
        -3.266407412190941e-01, -3.383295002935882e-01, -1.964237395967755e-01, 3.465429229977288e-02,
        2.500000000000001e-01, 3.518509343815957e-01, 2.939689006048397e-01, 1.026311318805893e-01,
        -1.352990250365493e-01, -3.118062532466679e-01, -3.467599613305369e-01, -2.242918965856590e-01,
    ],
    [
        2.500000000000000e-01, 2.242918965856591e-01, -6.897484482073575e-02, -3.118062532466678e-01,
        -3.266407412190941e-01, -1.026311318805894e-01, 1.964237395967755e-01, 3.518509343815957e-01,
        2.500000000000001e-01, -3.465429229977282e-02, -2.939689006048397e-01, -3.383295002935882e-01,
        -1.352990250365493e-01, 1.666639146194367e-01, 3.467599613305369e-01, 2.733004667504394e-01,
    ],
    [
        2.500000000000000e-01, 1.666639146194366e-01, -1.964237395967756e-01, -3.518509343815956e-01,
        -1.352990250365493e-01, 2.242918965856591e-01, 3.467599613305369e-01, 1.026311318805894e-01,
        -2.500000000000001e-01, -3.383295002935882e-01, -6.897484482073574e-02, 2.733004667504394e-01,
        3.266407412190941e-01, 3.465429229977289e-02, -2.939689006048397e-01, -3.118062532466677e-01,
    ],
    [
        2.500000000000000e-01, 1.026311318805894e-01, -2.939689006048397e-01, -2.733004667504393e-01,
        1.352990250365493e-01, 3.518509343815957e-01, 6.897484482073579e-02, -3.118062532466678e-01,
        -2.500000000000001e-01, 1.666639146194366e-01, 3.467599613305369e-01, 3.465429229977293e-02,
        -3.266407412190941e-01, -2.242918965856591e-01, 1.964237395967756e-01, 3.383295002935882e-01,
    ],
    [
        2.500000000000000e-01, 3.465429229977287e-02, -3.467599613305369e-01, -1.026311318805893e-01,
        3.266407412190941e-01, 1.666639146194366e-01, -2.939689006048397e-01, -2.242918965856591e-01,
        2.500000000000001e-01, 2.733004667504393e-01, -1.964237395967756e-01, -3.118062532466678e-01,
        1.352990250365493e-01, 3.383295002935882e-01, -6.897484482073578e-02, -3.518509343815956e-01,
    ],
    [
        2.500000000000000e-01, -3.465429229977287e-02, -3.467599613305369e-01, 1.026311318805893e-01,
        3.266407412190941e-01, -1.666639146194366e-01, -2.939689006048397e-01, 2.242918965856591e-01,
        2.500000000000001e-01, -2.733004667504393e-01, -1.964237395967756e-01, 3.118062532466678e-01,
        1.352990250365493e-01, -3.383295002935882e-01, -6.897484482073578e-02, 3.518509343815956e-01,
    ],
    [
        2.500000000000000e-01, -1.026311318805894e-01, -2.939689006048397e-01, 2.733004667504393e-01,
        1.352990250365493e-01, -3.518509343815957e-01, 6.897484482073579e-02, 3.118062532466678e-01,
        -2.500000000000001e-01, -1.666639146194366e-01, 3.467599613305369e-01, -3.465429229977293e-02,
        -3.266407412190941e-01, 2.242918965856591e-01, 1.964237395967756e-01, -3.383295002935882e-01,
    ],
    [
        2.500000000000000e-01, -1.666639146194366e-01, -1.964237395967756e-01, 3.518509343815956e-01,
        -1.352990250365493e-01, -2.242918965856591e-01, 3.467599613305369e-01, -1.026311318805894e-01,
        -2.500000000000001e-01, 3.383295002935882e-01, -6.897484482073574e-02, -2.733004667504394e-01,
        3.266407412190941e-01, -3.465429229977289e-02, -2.939689006048397e-01, 3.118062532466677e-01,
    ],
    [
        2.500000000000000e-01, -2.242918965856591e-01, -6.897484482073575e-02, 3.118062532466678e-01,
        -3.266407412190941e-01, 1.026311318805894e-01, 1.964237395967755e-01, -3.518509343815957e-01,
        2.500000000000001e-01, 3.465429229977282e-02, -2.939689006048397e-01, 3.383295002935882e-01,
        -1.352990250365493e-01, -1.666639146194367e-01, 3.467599613305369e-01, -2.733004667504394e-01,
    ],
    [
        2.500000000000000e-01, -2.733004667504394e-01, 6.897484482073575e-02, 1.666639146194366e-01,
        -3.266407412190941e-01, 3.383295002935882e-01, -1.964237395967755e-01, -3.465429229977288e-02,
        2.500000000000001e-01, -3.518509343815957e-01, 2.939689006048397e-01, -1.026311318805893e-01,
        -1.352990250365493e-01, 3.118062532466679e-01, -3.467599613305369e-01, 2.242918965856590e-01,
    ],
    [
        2.500000000000000e-01, -3.118062532466678e-01, 1.964237395967756e-01, -3.465429229977286e-02,
        -1.352990250365493e-01, 2.733004667504394e-01, -3.467599613305369e-01, 3.383295002935882e-01,
        -2.500000000000001e-01, 1.026311318805894e-01, 6.897484482073574e-02, -2.242918965856590e-01,
        3.266407412190941e-01, -3.518509343815957e-01, 2.939689006048397e-01, -1.666639146194367e-01,
    ],
    [
        2.500000000000000e-01, -3.383295002935882e-01, 2.939689006048397e-01, -2.242918965856591e-01,
        1.352990250365493e-01, -3.465429229977286e-02, -6.897484482073579e-02, 1.666639146194366e-01,
        -2.500000000000001e-01, 3.118062532466678e-01, -3.467599613305369e-01, 3.518509343815956e-01,
        -3.266407412190941e-01, 2.733004667504394e-01, -1.964237395967756e-01, 1.026311318805893e-01,
    ],
    [
        2.500000000000000e-01, -3.518509343815957e-01, 3.467599613305369e-01, -3.383295002935882e-01,
        3.266407412190941e-01, -3.118062532466678e-01, 2.939689006048397e-01, -2.733004667504394e-01,
        2.500000000000001e-01, -2.242918965856591e-01, 1.964237395967756e-01, -1.666639146194367e-01,
        1.352990250365493e-01, -1.026311318805893e-01, 6.897484482073578e-02, -3.465429229977293e-02,
    ],
];
