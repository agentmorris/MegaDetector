<img src="images/md_logo_450px.png" width="450"/><br/>

...helping conservation biologists spend less time doing boring things with camera trap images.

## Table of contents

1. [What's MegaDetector all about?](#whats-megadetector-all-about)
3. [How do I get started with MegaDetector?](#how-do-i-get-started-with-megadetector)
4. [Who is using MegaDetector?](#who-is-using-megadetector)
5. [Repo contents](#repo-contents)
6. [Contact](#contact)
7. [Gratuitous camera trap picture](#gratuitous-camera-trap-picture)
8. [License](#license)
9. [Contributing](#contributing)


## What's MegaDetector all about?

[MegaDetector](megadetector.md) is an AI model that identifies animals, people, and vehicles in camera trap images (which also makes it useful for eliminating blank images).  This model is trained on several million images from a variety of ecosystems.

MegaDetector only finds animals, it doesn't identify them to species level.  If you're looking for a species classifier, check out [SpeciesNet](https://github.com/google/cameratrapai), a species classifier that plays nicely with MegaDetector.

Here's a &ldquo;teaser&rdquo; image of what MegaDetector output looks like:

![Red bounding box on fox](images/detector_example.jpg)<br/>Image credit University of Washington.


## How do I get started with MegaDetector?

* If you are looking for a convenient tool to run MegaDetector, you don't need anything from this repository: check out [AddaxAI](https://addaxdatascience.com/addaxai/) (formerly EcoAssist), a GUI-based tool for running AI models (including MegaDetector) on camera trap images.
* If you're just <i>considering</i> the use of AI in your workflow, and you aren't even sure yet whether MegaDetector would be useful to you, we recommend reading the "[getting started with MegaDetector](getting-started.md)" page.
* If you're already familiar with MegaDetector and you're ready to run it on your data, see the [MegaDetector User Guide](megadetector.md) for instructions on running MegaDetector.
* If you're a programmer-type looking to use tools from this repo, check out the [MegaDetector Python package](https://pypi.org/project/megadetector/) that provides access to everything in this repo (yes, you guessed it, "pip install megadetector").
* If you have any questions, or you want to tell us that MegaDetector was amazing/terrible on your images, or you have a zillion images and you want some help digging out of that backlog, <a href="mailto:cameratraps@lila.science">email us</a>!

MegaDetector is just one of many tools that aims to make conservation biologists more efficient with AI.  If you want to learn about other ways to use AI to accelerate camera trap workflows, check out our of the field, affectionately titled &ldquo;[Everything I know about machine learning and camera traps](https://agentmorris.github.io/camera-trap-ml-survey/)&rdquo;.


## Who is using MegaDetector?

We work with ecologists all over the world to help them spend less time annotating images and more time thinking about conservation.  You can read a little more about how this works on our [getting started with MegaDetector](getting-started.md) page.

Here are a few of the organizations that have used MegaDetector... we're only listing organizations who (a) we know about and (b) have given us permission to refer to them here (or have posted publicly about their use of MegaDetector), so if you're using MegaDetector or other tools from this repo and would like to be added to this list, <a href="mailto:cameratraps@lila.science">email us</a>!

* [Arizona Department of Environmental Quality](http://azdeq.gov/)
* [Biometrio.earth](https://biometrio.earth/)
* [Blackbird Environmental](https://blackbirdenv.com/)
* [Camelot](https://camelotproject.org/)
* [Canadian Parks and Wilderness Society (CPAWS) Northern Alberta Chapter](https://cpawsnab.org/)
* [Conservation X Labs](https://conservationxlabs.com/)
* [Czech University of Life Sciences Prague](https://www.czu.cz/en)
* [Dudek Camera Trap AI Image Toolkit (AIT)](https://ait.dudek.com/)
* [EcoLogic Consultants Ltd.](https://www.consult-ecologic.com/)
* [Estación Biológica de Doñana](https://www.ebd.csic.es/)
* [Indigenous Desert Alliance](https://www.indigenousdesertalliance.com/)
* [Myall Lakes Dingo Project](https://carnivorecoexistence.info/myall-lakes-dingo-project/)
* [Norwegian Institute for Nature Research](https://www.nina.no/english/Home)
* [Okala](https://www.okala.io/)
* [Point No Point Treaty Council](https://pnptc.org/)
* [Ramat Hanadiv Nature Park](https://www.ramat-hanadiv.org.il/en/)
* [SPEA (Portuguese Society for the Study of Birds)](https://spea.pt/en/)
* [Sky Island Alliance](https://skyislandalliance.org/)
* [Synthetaic](https://www.synthetaic.com/)
* [Taronga Conservation Society](https://taronga.org.au/)
* [The Nature Conservancy in Wyoming](https://www.nature.org/en-us/about-us/where-we-work/united-states/wyoming/)
* [TrapTagger](https://wildeyeconservation.org/trap-tagger-about/)
* [University of California Davis Natural Reserves](https://naturalreserves.ucdavis.edu/)
* [Upper Yellowstone Watershed Group](https://www.upperyellowstone.org/)
* [Zamba Cloud](https://www.zambacloud.com/)
* [Parc national du Mont-Tremblant](https://www.sepaq.com/pq/mot/index.dot?language_id=1)
* [The Land Banking Group](https://thelandbankinggroup.com/)
* [New Zealand Department of Conservation](https://www.doc.govt.nz)
* [Habitat NZ](https://habitatnz.co.nz/)
* [Research Institute of Organic Agriculture](https://www.fibl.org/en/) (FiBL)
* [A/Vian Ecological Consulting](https://avianeco.com/)
* [Wildlife Insights](https://www.wildlifeinsights.org/)
* [MammalWeb](https://www.mammalweb.org/)

* [Applied Conservation Macro Ecology Lab](http://www.acmelab.ca/), University of Victoria
* [Banff National Park Resource Conservation](https://www.pc.gc.ca/en/pn-np/ab/banff/nature/conservation), Parks Canada
* [Blumstein Lab](https://blumsteinlab.eeb.ucla.edu/), UCLA
* [Borderlands Research Institute](https://bri.sulross.edu/), Sul Ross State University
* [Capitol Reef National Park](https://www.nps.gov/care/index.htm) / Utah Valley University
* [Canyon Critters Project](https://www.zooniverse.org/projects/arw36/canyon-critters), University of Georgia
* [Center for Biodiversity and Conservation](https://www.amnh.org/research/center-for-biodiversity-conservation), American Museum of Natural History
* [Centre for Ecosystem Science](https://www.unsw.edu.au/research/), UNSW Sydney
* [Cross-Cultural Ecology Lab](https://crossculturalecology.net/), Macquarie University
* [DC Cat Count](https://hub.dccatcount.org/), led by the Humane Rescue Alliance
* [Department of Anthropology and Archaeology](https://arts.ucalgary.ca/anthropology-archaeology), University of Calgary
* [Department of Fish and Wildlife Sciences](https://www.uidaho.edu/cnr/departments/fish-and-wildlife-sciences), University of Idaho
* [Department of Society & Conservation](https://www.umt.edu/environment/about/departments/socon/), W.A. Franke College of Forestry & Conservation, University of Montana
* [Department of Wildlife Ecology and Conservation](https://wec.ifas.ufl.edu/), University of Florida
* [Fodrie Lab](https://fodriefishecol.wixsite.com/unc-fish/), Institute of Marine Sciences, UNC Chapel Hill
* [Gola Forest Programme](https://www.rspb.org.uk/our-work/conservation/projects/scientific-support-for-the-gola-forest-programme/), Royal Society for the Protection of Birds (RSPB)
* [Graeme Shannon's Research Group](https://wildliferesearch.co.uk/group-1), Bangor University 
* [Grizzly Bear Recovery Program](https://www.fws.gov/office/grizzly-bear-recovery-program), U.S. Fish & Wildlife Service
* [Hamaarag](https://hamaarag.org.il/), The Steinhardt Museum of Natural History, Tel Aviv University
* [Institut des Science de la Forêt Tempérée](https://isfort.uqo.ca/) (ISFORT), Université du Québec en Outaouais
* [Lab of Dr. Bilal Habib](https://bhlab.in/about), the Wildlife Institute of India
* [Landscape Ecology Lab](https://www.concordia.ca/artsci/geography-planning-environment/research/labs/lel.html), Concordia University
* [Mammal Spatial Ecology and Conservation Lab](https://labs.wsu.edu/dthornton/), Washington State University
* [McLoughlin Lab in Population Ecology](http://mcloughlinlab.ca/lab/), University of Saskatchewan
* [National Wildlife Refuge System, Southwest Region](https://www.fws.gov/about/region/southwest), U.S. Fish & Wildlife Service
* [Northern Great Plains Program](https://nationalzoo.si.edu/news/restoring-americas-prairie), Smithsonian
* [Polar Ecology Group](https://polarecologygroup.wordpress.com), University of Gdansk
* [Quantitative Ecology Lab](https://depts.washington.edu/sefsqel/), University of Washington
* [San Diego Field Station](https://www.usgs.gov/centers/werc/science/san-diego-field-station), U.S. Geological Survey
* [Santa Monica Mountains Recreation Area](https://www.nps.gov/samo/index.htm), National Park Service
* [Seattle Urban Carnivore Project](https://www.zoo.org/seattlecarnivores), Woodland Park Zoo
* [Serra dos Órgãos National Park](https://www.icmbio.gov.br/parnaserradosorgaos/), ICMBio
* [Snapshot USA](https://www.snapshot-usa.org/), Smithsonian
* [TROPECOLNET project](https://www.anabenitezlopez.com/research/global-change-biology/tropecolnet/), Museo Nacional de Ciencias Naturales
* [Wildlife Coexistence Lab](https://wildlife.forestry.ubc.ca/), University of British Columbia
* [Wildlife Research](https://www.dfw.state.or.us/wildlife/research/index.asp), Oregon Department of Fish and Wildlife
* [Kohl Wildlife Lab](https://kohlwildlifelab.com/), University of Georgia
* [SPEC Lab](https://thespeclab.weebly.com/) and [Cherry Lab](https://www.ckwri.tamuk.edu/about/michael-j-cherry-phd), Caesar Kleberg Wildlife Research Institute, Texas A&M Kingsville
* Ecology and Conservation of Amazonian Vertebrates Research Group, Federal University of Amapá
* Department of Ecology, TU Berlin
* Ghost Cat Analytics
* Protected Areas Unit, Canadian Wildlife Service
* Conservation and Restoration Science Branch, New South Wales Department of Climate Change, Energy, the Environment and Water

* [School of Natural Sciences](https://www.utas.edu.au/natural-sciences), University of Tasmania ([story](https://www.utas.edu.au/about/news-and-stories/articles/2022/1204-innovative-camera-network-keeps-close-eye-on-tassie-wildlife))
* [Kenai National Wildlife Refuge](https://www.fws.gov/refuge/kenai), U.S. Fish & Wildlife Service ([story](https://www.peninsulaclarion.com/sports/refuge-notebook-new-technology-increases-efficiency-of-refuge-cameras/))

* [Idaho Department of Fish and Game](https://idfg.idaho.gov/) ([fancy PBS video](https://www.youtube.com/watch?v=uEsL8EZKpbA&t=261s&ab_channel=OutdoorIdaho))
* [Australian Wildlife Conservancy](https://www.australianwildlife.org/) (blog posts [1](https://www.australianwildlife.org/cutting-edge-technology-delivering-efficiency-gains-in-conservation/), [2](https://www.australianwildlife.org/efficiency-gains-at-the-cutting-edge-of-technology/), [3](https://www.australianwildlife.org/federal-grant-to-fund-ai-supported-wildlife-recognisers))
* [Bavarian Forest National Park](https://www.nationalpark-bayerischer-wald.bayern.de/english/index.htm) ([story](https://customers.microsoft.com/en-au/story/1667539539271247797-nationalparkbayerischerwald-azure-en))
* [Environs Kimberley](https://www.environskimberley.org.au) ([blog post](https://www.environskimberley.org.au/ai_megadetector))
* [Felidae Conservation Fund](https://felidaefund.org/) ([WildePod platform](https://wildepod.org/)) ([blog post](https://abhaykashyap.com/blog/ai-powered-camera-trap-image-annotation-system/))
* [Island Conservation](https://www.islandconservation.org/) (blog posts [1](https://www.islandconservation.org/conservation-machine-learning/),[2](https://news.lenovo.com/island-conservation-machine-learning-solutions-nvidia-island-ecosystems/?sprinklrid=12869857824&linkId=356951919)) ([video](https://www.lenovo.com/content/dam/lenovo/iso/customer-references-coe/one-lenovo-customer-stories/wfh/videos/WFH-One-Lenovo-ENG-subtitles.mp4))
* [Alberta Biodiversity Monitoring Institute (ABMI)](https://www.abmi.ca/home.html) ([WildTrax platform](https://www.wildtrax.ca/)) (blog posts [1](https://wildcams.ca/blog/the-abmi-visits-the-zoo/),[2](http://blog.abmi.ca/2023/06/14/making-wildtrax-its-not-a-kind-of-magic-behind-the-screen/))
* [Shan Shui Conservation Center](http://en.shanshui.org/) ([blog post](https://mp.weixin.qq.com/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs)) ([translated blog post](https://mp-weixin-qq-com.translate.goog/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs&_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp)) ([Web demo](https://cameratrap-ai.hinature.cn/home))
* [Irvine Ranch Conservancy](http://www.irconservancy.org/) ([story](https://www.ocregister.com/2022/03/30/ai-software-is-helping-researchers-focus-on-learning-about-ocs-wild-animals/))
* [Wildlife Protection Solutions](https://wildlifeprotectionsolutions.org/) ([story](https://customers.microsoft.com/en-us/story/1384184517929343083-wildlife-protection-solutions-nonprofit-ai-for-earth), [story](https://www.enterpriseai.news/2023/02/20/ai-helps-wildlife-protection-solutions-safeguard-endangered-species/))
* [Q42](https://www.q42.nl/en) ([blog post](https://engineering.q42.nl/ai-bear-repeller/))
* [Agouti](https://agouti.eu/) ([report](https://efsa.onlinelibrary.wiley.com/doi/pdf/10.2903/sp.efsa.2023.EN-8217))
* [Trapper](https://trapper-project.readthedocs.io/en/latest/overview.html) ([tutorial](https://trapper-project.readthedocs.io/en/latest/tutorial.html))
* [BirdLife Malta](https://birdlifemalta.org/) ([tweet](https://x.com/BirdLife_Malta/status/1817456839862173783?t=S-KRiZ5R1-CoW8-tbYNjqQ&s=03)) ([LI post](https://www.linkedin.com/posts/birdlifemalta_worldnatureconservationday-shearwater-colony-activity-7223220656589463553-X2Mc/?utm_source=share&utm_medium=member_android))
* [Endangered Landscapes and Seascapes Programme](https://www.endangeredlandscapes.org/), Cambridge Conservation Initiative ([blog post](https://www.endangeredlandscapes.org/news/ai-for-wildlife-monitoring-a-real-time-alert-system-for-bears-and-wild-boars-in-romanias-carpathian-mountains/))
* [UK National Hedgehog Monitoring Programme](https://www.the-ies.org/analysis/lens-wild-innovations-wildlife-0) ([blog post](https://www.the-ies.org/analysis/lens-wild-innovations-wildlife-0))
* [Fort Collins Science Center](https://www.usgs.gov/centers/fort-collins-science-center/), U.S. Geological Survey ([blog post](https://www.usgs.gov/centers/fort-collins-science-center/science/early-detection-and-rapid-response-tools-invasive-tegu))
* [Ontario Ministry of Natural Resources](https://www.ontario.ca/page/ministry-natural-resources) ([post](https://www.facebook.com/ONresources/posts/artificial-intelligence-ai-to-the-rescueone-disadvantage-of-trail-cameras-and-ca/1012916944364710/))
* [Loisaba Conservancy](https://www.loisaba.com/) ([blog post](https://www.loisaba.com/post/help-scientists-understand-how-an-invasive-cactus-affects-mammal-behaviour))
* [Urban Rivers](https://urbanrivers.org/) ([blog post](https://www.morescode-analytics.com/2025/06/27/filtering-90-pct-blank-wildlife.html))
* [Road Ecology Center](https://roadecology.ucdavis.edu/), University of California, Davis ([Wildlife Observer Network platform](https://roadecology.ucdavis.edu/research/projects/wildlife-observer-network))
* [The Nature Conservancy in California](https://www.nature.org/en-us/about-us/where-we-work/united-states/california/) ([Animl platform](https://github.com/tnc-ca-geo/animl-frontend)) ([story](https://www.vision-systems.com/non-factory/environment-agriculture/article/14304433/the-nature-conservancy-brings-cameras-ai-to-invasive-species-prevention))
* [San Diego Zoo Wildlife Alliance](https://science.sandiegozoo.org/)  ([Animl R package](https://github.com/conservationtechlab/animl))
* [TerrOïko](https://www.terroiko.fr/) ([OCAPI platform](https://www.terroiko.fr/ocapi))
* [Wildlife Division](https://www.michigan.gov/dnr/about/contact/wildlife), Michigan Department of Natural Resources ([blog post](https://www.michigan.gov/dnr/about/newsroom/releases/2025/08/18/dnr-researchers-to-test-trail-cameras-in-elk-survey))

Also see:

* The [list of MD-related GUIs, platforms, and GitHub repos](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#is-there-a-gui) within the MegaDetector User Guide... although you can never have too many lists, so here they are in a concise comma-separated list: [Wildlife Insights](https://wildlifeinsights.org/), [Animal Detect](https://www.animaldetect.com), [TrapTagger](https://wildeyeconservation.org/trap-tagger-about/), [WildTrax](https://www.wildtrax.ca/), [Agouti](https://agouti.eu/), [Trapper](https://trapper-project.readthedocs.io/en/latest/overview.html), [Camelot](https://camelotproject.org/), [WildePod](https://wildepod.org/), [wpsWatch](https://wildlabs.net/inventory/products/wpswatch), [TNC Animl](https://animl.camera/) ([code](https://github.com/tnc-ca-geo/animl-frontend)), [Wildlife Observer Network](https://roadecology.ucdavis.edu/research/projects/wildlife-observer-network), [Zooniverse ML Subject Assistant](https://subject-assistant.zooniverse.org/#/intro), [Dudek AI Image Toolkit](https://ait.dudek.com), [Zamba Cloud](https://github.com/drivendataorg/zamba), [OCAPI](https://www.terroiko.fr/ocapi/), [BoquilaHUB](https://boquila.org/hub)

* [Peter's map of AddaxAI (formerly EcoAssist) users](https://addaxdatascience.com/addaxai/) (who are also MegaDetector users!)

* The list of papers tagged "MegaDetector" on our [list of papers about ML and camera traps](https://agentmorris.github.io/camera-trap-ml-survey/#camera-trap-ml-papers)


## Repo contents

### Repo history

MegaDetector was initially developed by the [Microsoft AI for Earth program](https://www.microsoft.com/en-us/ai/ai-for-earth); this repo was forked from the microsoft/cameratraps repo and is maintained by the original MegaDetector developers (who are no longer at Microsoft, but are absolutely fantastically eternally grateful to Microsoft for the investment and commitment that made MegaDetector happen).  If you're interested in MD's history, see the [downloading the model](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#downloading-the-model) section in the MegaDetector User Guide to learn about the history of MegaDetector releases, and the [can you share the training data?](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#can-you-share-the-training-data) section to learn about the training data used in each of those releases.

### What the code in this repo does

The core functionality provided in this repo is:

- Tools for training and running [MegaDetector](megadetector.md).
- Tools for working with MegaDetector output, e.g. for reviewing the results of a large processing batch.
- Tools to convert among frequently-used camera trap metadata formats.

This repo does not host the data used to train MegaDetector, but we work with our collaborators to make data and annotations available whenever possible on [lila.science](http://lila.science).  See the [MegaDetector training data](megadetector.md#can-you-share-the-training-data) section to learn more about the data used to train MegaDetector.

### Repo organization

This repo is organized into the following folders...

#### megadetector/detection

Code for running models, especially MegaDetector.


##### megadetector/postprocessing

Code for common operations one might do after running MegaDetector, e.g. [generating preview pages to summarize your results](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py), [separating images into different folders based on AI results](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/separate_detections_into_folders.py), or [converting results to a different format](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/convert_output_format.py).


#### megadetector/utils

Small utility functions for string manipulation, filename manipulation, downloading files from URLs, etc.


#### megadetector/visualization

Tools for visualizing images with ground truth and/or predicted bounding boxes.


#### megadetector/data_management

Code for:

* Converting frequently-used metadata formats to [COCO Camera Traps](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md#coco-cameratraps-format) format
* Converting the output of AI models (especially YOLOv5) to the format used for AI results throughout this repo
* Creating, visualizing, and  editing COCO Camera Traps .json databases


#### megadetector/api

Code for hosting our models as an API, either for synchronous operation (i.e., for real-time inference) or as a batch process (for large biodiversity surveys).  This folder is largely deprecated, and primarily exists now only to hold documentation that is still relevant, to which there are permanent links.  Mostly everything in this folder has been moved to the "archive" folder.


#### megadetector/classification

This folder is largely deprecated thanks to the release of [SpeciesNet](https://github.com/google/cameratrapai), a species classifier that is better than any of the classifiers we ever trained with the stuff in this folder.  That said, this folder contains code for training species classifiers on new data sets, generally trained on MegaDetector crops.

Here's another "teaser image" of what you get at the end of training and running a classifier:

<img src="images/warthog_classifications.jpg" width="700"><br/>Image credit University of Minnesota, from the Snapshot Safari program.


#### megadetector/taxonomy_mapping

Code to facilitate mapping data-set-specific category names (e.g. "lion", which means very different things in Idaho vs. South Africa) to a standard taxonomy.


#### envs

Environment files... specifically .yml files for mamba/conda environments (these are what we recommend in our [MegaDetector User Guide](megadetector.md)), and a requirements.txt for the pip-inclined.


#### images

Media used in documentation.


#### archive

Old code that we didn't <i>quite</i> want to delete, but is basically obsolete.


#### sandbox

Random things that don't fit in any other directory, but aren't quite deprecated.  Mostly postprocessing scripts that were built for a single use case but could potentially be useful in the future.


#### test_images

A handful of images from [LILA](https://lila.science) that facilitate testing and debugging.


## Contact

For questions about this repo, contact [cameratraps@lila.science](mailto:cameratraps@lila.science).

You can also chat with us and the broader camera trap AI community on the [AI for Conservation forum at WILDLABS](https://wildlabs.net/groups/ai-conservation) or the [AI for Conservation Slack group](https://aiforconservation.slack.com).


## Gratuitous camera trap picture

![Bird flying above water](images/nacti.jpg)<br/>Image credit USDA, from the [NACTI](http://lila.science/datasets/nacti) data set.

You will find lots more gratuitous camera trap pictures sprinkled about this repo.  It's like a scavenger hunt.


## License

This repository is licensed with the [MIT license](https://opensource.org/license/mit/).

Code written on or before April 28, 2023 is [copyright Microsoft](https://github.com/Microsoft/dotnet/blob/main/LICENSE).


## Contributing

This project welcomes contributions, as pull requests, issues, or suggestions by [email](mailto:cameratraps@lila.science).  We have a [list](https://dmorris.net/task-viewer/?file=https://raw.githubusercontent.com/agentmorris/MegaDetector/refs/heads/main/TODO.md) of issues that we're hoping to address, many of which would be good starting points for new contributors.  We also depend on other open-source tools that help users run MegaDetector (particularly [AddaxAI](https://github.com/PetervanLunteren/AddaxAI) (formerly EcoAssist), and open-source tools that help users work with MegaDetector results (particularly [Timelapse](https://github.com/saulgreenberg/Timelapse)).  If you are looking to get involved in GUI development, reach out to the developers of those tools as well!

If you are interesting in getting involved in the conservation technology space, and MegaDetector just happens to be the first page you landed on, and none of our open issues are getting you fired up, don't fret!  Head over to the [WILDLABS discussion forums](https://wildlabs.net/discussions) and let the community know you're a developer looking to get involved.  Someone needs your help!

Information about the coding conventions, linting, testing, and documentation tools used by this repo are available in [developers.md](developers.md).

Speaking of contributions... thanks to Erin Roche from Idaho Fish and Game for the MegaDetector logo at the top of this page!
