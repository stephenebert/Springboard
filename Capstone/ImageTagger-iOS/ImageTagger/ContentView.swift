import SwiftUI
import PhotosUI
import Vision
import CoreML
import UIKit

enum SortOption: String, CaseIterable {
    case confidence = "Confidence"
    case alphabetical = "A–Z"
}

struct ContentView: View {
    // MARK: - State
    @State private var pickerItem: PhotosPickerItem?
    @State private var imageData: Data?
    @State private var tags: [TagResult] = []
    @State private var topK: Int = 5
    @State private var isAnalyzing = false
    @State private var lastError: String?
    @State private var showSettings = false
    @State private var analysisTime: TimeInterval = 0
    @State private var imageSize: CGSize = .zero
    @State private var confidenceThreshold: Double = 0.05
    @State private var sortBy: SortOption = .confidence
    @State private var showingImageDetail = false

    // Focus/boost knobs
    @State private var focusTerms: [String] = [
        "denim","jean","shirt","jacket","dress","sneaker",
        "phone","hat","cap","sunglasses","bag"
    ]
    @State private var applyBoost: Bool = true

    private let tagLimitRange = 1...20
    private let confidenceRange = 0.0...1.0

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    headerSection
                    Divider()
                    imageCard
                    actionButtons
                    resultsSection
                    if !tags.isEmpty { analysisInfoSection }
                }
                .frame(maxWidth: 600)
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .frame(maxWidth: .infinity)
            }
            .navigationTitle("Image Tagger")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                settingsSheet
            }
            .fullScreenCover(isPresented: $showingImageDetail) {
                imageDetailView
            }
            .task(id: pickerItem) { await loadImage() }
            .alert("Analysis Failed", isPresented: Binding(
                get: { lastError != nil },
                set: { _ in lastError = nil }
            )) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(lastError ?? "Unknown error")
            }
        }
    }

    // MARK: - UI sections (same look you had)

    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Number of tags: \(topK)")
                    .font(.headline)
                Spacer()
                Stepper("", value: $topK, in: tagLimitRange)
                    .labelsHidden().frame(width: 120)
            }

            HStack {
                Text("Min confidence: \(Int(confidenceThreshold * 100))%")
                    .font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                Slider(value: $confidenceThreshold, in: confidenceRange, step: 0.05)
                    .frame(width: 140)
            }

            HStack {
                Text("Sort by:")
                    .font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                Picker("Sort by", selection: $sortBy) {
                    ForEach(SortOption.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).frame(width: 200)
            }
        }
    }

    private var imageCard: some View {
        Card {
            Group {
                if let data = imageData, let ui = UIImage(data: data) {
                    GeometryReader { geo in
                        Image(uiImage: ui)
                            .resizable()
                            .scaledToFill()
                            .frame(width: geo.size.width, height: 320)
                            .clipped()
                            .onTapGesture { showingImageDetail = true }
                            .onAppear { imageSize = ui.size }
                    }
                    .frame(height: 320)
                    .overlay(alignment: .topTrailing) {
                        Button { showingImageDetail = true } label: {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .foregroundStyle(.white)
                                .padding(8)
                                .background(.ultraThickMaterial, in: Circle())
                        }
                        .padding(12)
                    }
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 48)).foregroundStyle(.secondary)
                        VStack(spacing: 4) {
                            Text("No image selected")
                                .font(.headline).foregroundStyle(.secondary)
                            Text("Tap 'Select Image' to get started")
                                .font(.caption).foregroundStyle(.tertiary)
                        }
                    }
                    .frame(height: 200)
                }
            }
        }
    }

    private var actionButtons: some View {
        HStack(spacing: 12) {
            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Select Image", systemImage: "photo.on.rectangle")
                    .fontWeight(.semibold).frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered).controlSize(.large)

            Button {
                Task { await analyze() }
            } label: {
                HStack(spacing: 6) {
                    if isAnalyzing { ProgressView().scaleEffect(0.8) }
                    else { Image(systemName: "wand.and.rays") }
                    Text(isAnalyzing ? "Analyzing..." : "Analyze")
                }
                .fontWeight(.semibold).frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isAnalyzing || imageData == nil)
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Labels").font(.headline)
                if !tags.isEmpty {
                    Spacer()
                    Text("\(sortedTags.count) results")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            if !sortedTags.isEmpty {
                Card(insets: .init(top: 12, leading: 12, bottom: 12, trailing: 12)) {
                    TagFlow(spacing: 10, rowSpacing: 10) {
                        ForEach(sortedTags, id: \.id) { tag in
                            EnhancedChip(tag: tag)
                        }
                    }
                }
            } else if !tags.isEmpty {
                Card {
                    VStack(spacing: 8) {
                        Image(systemName: "slider.horizontal.below.rectangle")
                            .font(.title2).foregroundStyle(.secondary)
                        Text("No results match current filters")
                            .font(.subheadline).foregroundStyle(.secondary)
                        Text("Try lowering the confidence threshold")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity).padding()
                }
            }
        }
    }

    private var analysisInfoSection: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "info.circle").foregroundStyle(.blue)
                    Text("Analysis Info").font(.subheadline).fontWeight(.medium)
                    Spacer()
                }
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    GridRow {
                        Text("Processing time:").foregroundStyle(.secondary)
                        Text("\(String(format: "%.2f", analysisTime))s").fontWeight(.medium)
                    }
                    GridRow {
                        Text("Image size:").foregroundStyle(.secondary)
                        Text("\(Int(imageSize.width))×\(Int(imageSize.height))").fontWeight(.medium)
                    }
                    GridRow {
                        Text("Model:").foregroundStyle(.secondary)
                        Text("Vision → MobileNet fallback").fontWeight(.medium)
                    }
                }
                .font(.caption)
            }
        }
    }

    private var settingsSheet: some View {
        NavigationStack {
            Form {
                Section("Classification Settings") {
                    Stepper("Tag Limit: \(topK)", value: $topK, in: tagLimitRange)
                    HStack {
                        Text("Confidence Threshold")
                        Spacer()
                        Text("\(Int(confidenceThreshold * 100))%").foregroundStyle(.secondary)
                    }
                    Slider(value: $confidenceThreshold, in: 0...1, step: 0.01)
                    Toggle("Boost focus terms", isOn: $applyBoost)
                }
                Section("Focus Terms") {
                    TextField("Comma separated", text: Binding(
                        get: { focusTerms.joined(separator: ", ") },
                        set: { focusTerms = $0
                            .split(separator: ",")
                            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                            .filter { !$0.isEmpty }
                        }
                    ))
                }
                Section {
                    Button("Reset to Defaults") {
                        withAnimation {
                            topK = 5
                            confidenceThreshold = 0.05
                            sortBy = .confidence
                            focusTerms = ["denim","jean","shirt","jacket","dress","sneaker","phone","hat","cap","sunglasses","bag"]
                            applyBoost = true
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) {
                Button("Done") { showSettings = false }
            }}
        }
    }

    private var imageDetailView: some View {
        NavigationStack {
            ZoomableImageView(imageData: imageData)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { showingImageDetail = false }
                }}
        }
    }

    // MARK: - Derived

    private var filteredTags: [TagResult] {
        tags.filter { $0.confidence >= confidenceThreshold }
    }
    private var sortedTags: [TagResult] {
        let base = filteredTags.prefix(topK)
        switch sortBy {
        case .confidence:   return base.sorted { $0.confidence > $1.confidence }
        case .alphabetical: return base.sorted { $0.label < $1.label }
        }
    }

    // MARK: - Actions

    private func loadImage() async {
        guard let item = pickerItem else { return }
        if let data = try? await item.loadTransferable(type: Data.self) {
            await MainActor.run {
                withAnimation(.easeInOut) {
                    imageData = data
                    tags.removeAll()
                    analysisTime = 0
                }
            }
        }
    }

    private func analyze() async {
        guard !isAnalyzing else { return }
        guard let data = imageData, let ui = UIImage(data: data) else {
            lastError = "Please select an image first."
            return
        }

        await MainActor.run { isAnalyzing = true }
        let start = Date()

        defer { Task { @MainActor in isAnalyzing = false } }

        do {
            let results = try ZeroShotTagger.shared.predictTags(
                from: ui,
                topK: max(1, topK),
                threshold: max(0, min(1, confidenceThreshold)),
                focusTerms: focusTerms.map { $0.lowercased() },
                applyBoost: applyBoost
            )

            let duration = Date().timeIntervalSince(start)
            await MainActor.run {
                analysisTime = duration
                let h = UINotificationFeedbackGenerator()
                h.notificationOccurred(.success)
                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                    tags = results
                }
            }
        } catch {
            await MainActor.run {
                lastError = error.localizedDescription
                let h = UINotificationFeedbackGenerator()
                h.notificationOccurred(.error)
            }
        }
    }
}

// MARK: - Reusable UI bits 

private struct EnhancedChip: View {
    let tag: TagResult
    private var confidenceColor: Color {
        switch tag.confidence {
        case 0.8...1.0: return .green
        case 0.5..<0.8: return .orange
        default: return .red
        }
    }
    var body: some View {
        HStack(spacing: 6) {
            Text(tag.label).font(.callout).fontWeight(.medium)
            Text("\(Int(tag.confidence * 100))%")
                .font(.caption).fontWeight(.semibold)
                .foregroundStyle(confidenceColor)
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(confidenceColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
        }
        .foregroundStyle(.primary)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().stroke(.black.opacity(0.08), lineWidth: 0.5))
    }
}

private struct Card<Content: View>: View {
    var insets: EdgeInsets = EdgeInsets(top: 16, leading: 16, bottom: 16, trailing: 16)
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .padding(insets)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(.black.opacity(0.05), lineWidth: 0.5))
            .shadow(color: .black.opacity(0.08), radius: 12, y: 4)
    }
}

/// Flow layout for chips
private struct TagFlow: Layout {
    var spacing: CGFloat = 8
    var rowSpacing: CGFloat = 8
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0
        for s in subviews {
            let sz = s.sizeThatFits(.unspecified)
            if x + sz.width > maxWidth, x > 0 { x = 0; y += rowH + rowSpacing; rowH = 0 }
            rowH = max(rowH, sz.height)
            x += sz.width + spacing
        }
        return CGSize(width: maxWidth.isFinite ? maxWidth : x, height: y + rowH)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0, maxX = bounds.maxX
        for s in subviews {
            let sz = s.sizeThatFits(.unspecified)
            if x + sz.width > maxX, x > bounds.minX { x = bounds.minX; y += rowH + rowSpacing; rowH = 0 }
            s.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(sz))
            x += sz.width + spacing
            rowH = max(rowH, sz.height)
        }
    }
}

private struct ZoomableImageView: View {
    let imageData: Data?
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastScale: CGFloat = 1.0
    @State private var lastOffset: CGSize = .zero
    var body: some View {
        GeometryReader { _ in
            if let data = imageData, let ui = UIImage(data: data) {
                Image(uiImage: ui)
                    .resizable().scaledToFit()
                    .scaleEffect(scale).offset(offset)
                    .gesture(
                        SimultaneousGesture(
                            MagnificationGesture()
                                .onChanged { v in scale = lastScale * v }
                                .onEnded { _ in
                                    lastScale = min(max(scale, 1.0), 3.0)
                                    withAnimation(.spring()) { scale = lastScale }
                                    if lastScale == 1.0 { offset = .zero; lastOffset = .zero }
                                },
                            DragGesture()
                                .onChanged { value in
                                    offset = CGSize(width: lastOffset.width + value.translation.width,
                                                    height: lastOffset.height + value.translation.height)
                                }
                                .onEnded { _ in lastOffset = offset }
                        )
                    )
                    .onTapGesture(count: 2) {
                        withAnimation(.spring()) {
                            if scale > 1.0 { scale = 1.0; offset = .zero }
                            else { scale = 2.0 }
                            lastScale = scale; lastOffset = offset
                        }
                    }
            }
        }
        .clipped().background(Color.black)
    }
}
